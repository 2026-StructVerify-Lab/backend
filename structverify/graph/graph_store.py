"""
graph/graph_store.py — Graph DB 인터페이스 (Neo4j)

Claim Graph / Evidence Graph / Temporal Graph 의 노드·엣지를 Neo4j에 저장하고
서브그래프를 조회한다.

[설계 — 라이브러리 배포 전제]
StructVerify는 라이브러리로 배포되므로 Neo4j는 *완전 옵셔널*이다.
  - 사용자가 default.yaml의 graph.store.enabled=true + 접속정보를 넣어야 동작
  - enabled=false / 섹션 없음 → GraphStore 비활성, 모든 호출 no-op
  - neo4j 패키지 미설치 → 경고 후 비활성 (파이프라인은 그대로 진행)
  - 연결/쿼리 실패 → 경고 후 해당 호출만 skip (검증 파이프라인 중단 없음)

default.yaml 예시:
    graph:
      store:
        enabled: true
        uri: bolt://localhost:7687
        user: neo4j
        password: ${NEO4J_PASSWORD}
        database: neo4j          # 선택, 기본 neo4j

[참고] GraphRAG (arXiv 2501.00309)
  Graph 기반 Evidence 검색 — subgraph 조회를 통한 다중 hop 추론 지원
"""
from __future__ import annotations

from structverify.core.schemas import GraphNode, GraphEdge
from structverify.utils.logger import get_logger

logger = get_logger(__name__)


def _safe_label(raw: str) -> str:
    """Cypher 라벨/관계타입에 안전한 문자열로 정제.

    영숫자와 밑줄만 허용 (Cypher injection 방지). node_type/edge_type은
    Enum value라 보통 안전하지만 방어적으로 정제한다.
    """
    cleaned = "".join(
        ch if (ch.isalnum() or ch == "_") else "_" for ch in str(raw)
    )
    if not cleaned or cleaned[0].isdigit():
        cleaned = "N_" + cleaned
    return cleaned


def _to_primitive(value):
    """Neo4j property로 쓸 수 있게 값을 정규화.

    Neo4j는 중첩 dict/list를 property로 못 받으므로, 복합 타입은
    문자열로 직렬화한다. None/숫자/bool/str은 그대로 통과.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, (str, int, float, bool)) for v in value):
            return list(value)
        return str(list(value))
    return str(value)


def _clean_props(props: dict | None) -> dict:
    """properties dict의 모든 값을 Neo4j 호환 형태로 변환."""
    if not props:
        return {}
    return {k: _to_primitive(v) for k, v in props.items()}


class GraphStore:
    """
    Neo4j 기반 Graph DB 인터페이스.

    노드/엣지를 Cypher MERGE로 upsert하고, N-hop 서브그래프를 조회한다.
    Neo4j가 없거나 비활성화면 모든 메서드가 안전하게 no-op 한다.
    """

    def __init__(self, config: dict | None = None):
        """
        Args:
            config: default.yaml의 graph.store 섹션
                    {enabled, uri, user, password, database}
        """
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.database = self.config.get("database") or "neo4j"
        self.driver = None
        self._connect_failed = False

        if not self.enabled:
            logger.info(
                "[GraphStore] graph.store.enabled=false — Neo4j 저장 비활성화 "
                "(노드/엣지는 메모리에서만 사용)"
            )
            return

        uri = self.config.get("uri")
        user = self.config.get("user")
        password = self.config.get("password")
        if not uri:
            logger.warning(
                "[GraphStore] enabled=true 이지만 graph.store.uri 미설정 "
                "— Neo4j 저장 skip"
            )
            self.enabled = False
            return

        # neo4j 패키지는 옵셔널 의존성 — 미설치 시 graceful degrade
        try:
            from neo4j import AsyncGraphDatabase
        except ImportError:
            logger.warning(
                "[GraphStore] 'neo4j' 패키지 미설치 — Neo4j 저장 skip. "
                "사용하려면: pip install neo4j"
            )
            self.enabled = False
            return

        try:
            auth = (user, password) if user else None
            self.driver = AsyncGraphDatabase.driver(uri, auth=auth)
            logger.info(
                f"[GraphStore] Neo4j driver 초기화: {uri} "
                f"(database={self.database})"
            )
        except Exception as e:
            logger.warning(
                f"[GraphStore] Neo4j driver 초기화 실패 — 저장 skip: {e}"
            )
            self.enabled = False
            self.driver = None

    def is_active(self) -> bool:
        """저장이 실제로 가능한 상태인지."""
        return bool(
            self.enabled and self.driver is not None
            and not self._connect_failed
        )

    async def save_nodes(self, nodes: list[GraphNode]) -> int:
        """
        노드 리스트를 Neo4j에 upsert한다 (Cypher MERGE).

        node_type을 라벨로, node_id를 고유 키로 사용.
        Returns: 저장된 노드 수 (비활성/실패 시 0).
        """
        if not self.is_active() or not nodes:
            return 0
        # 라벨(node_type)별로 묶어서 UNWIND 일괄 처리
        by_label: dict[str, list[dict]] = {}
        for n in nodes:
            label = (
                n.node_type.value if hasattr(n.node_type, "value")
                else str(n.node_type)
            )
            label = _safe_label(label)
            by_label.setdefault(label, []).append({
                "node_id": n.node_id,
                "label": n.label,
                "domain": getattr(n, "domain", None),
                "props": _clean_props(getattr(n, "properties", None)),
            })

        saved = 0
        try:
            async with self.driver.session(database=self.database) as session:
                for label, rows in by_label.items():
                    query = (
                        f"UNWIND $rows AS row "
                        f"MERGE (n:{label} {{node_id: row.node_id}}) "
                        f"SET n.label = row.label, "
                        f"    n.domain = row.domain, "
                        f"    n += row.props"
                    )
                    await session.run(query, rows=rows)
                    saved += len(rows)
            logger.info(
                f"[GraphStore] 노드 {saved}건 저장 (labels={list(by_label)})"
            )
        except Exception as e:
            logger.warning(f"[GraphStore] 노드 저장 실패 — skip: {e}")
            self._connect_failed = True
            return 0
        return saved

    async def save_edges(self, edges: list[GraphEdge]) -> int:
        """
        엣지 리스트를 Neo4j에 upsert한다 (Cypher MATCH + MERGE).

        from_node/to_node가 이미 저장된 노드를 가리킨다고 가정.
        Returns: 저장된 엣지 수 (비활성/실패 시 0).
        """
        if not self.is_active() or not edges:
            return 0
        by_type: dict[str, list[dict]] = {}
        for e in edges:
            etype = (
                e.edge_type.value if hasattr(e.edge_type, "value")
                else str(e.edge_type)
            )
            etype = _safe_label(etype).upper()
            by_type.setdefault(etype, []).append({
                "from": e.from_node,
                "to": e.to_node,
                "weight": getattr(e, "weight", 1.0),
                "props": _clean_props(getattr(e, "properties", None)),
            })

        saved = 0
        try:
            async with self.driver.session(database=self.database) as session:
                for etype, rows in by_type.items():
                    # 노드는 라벨 무관하게 node_id로 매칭
                    query = (
                        f"UNWIND $rows AS row "
                        f"MATCH (a {{node_id: row.from}}) "
                        f"MATCH (b {{node_id: row.to}}) "
                        f"MERGE (a)-[r:{etype}]->(b) "
                        f"SET r.weight = row.weight, r += row.props"
                    )
                    await session.run(query, rows=rows)
                    saved += len(rows)
            logger.info(
                f"[GraphStore] 엣지 {saved}건 저장 (types={list(by_type)})"
            )
        except Exception as e:
            logger.warning(f"[GraphStore] 엣지 저장 실패 — skip: {e}")
            self._connect_failed = True
            return 0
        return saved

    async def save_graph(
        self, nodes: list[GraphNode], edges: list[GraphEdge],
    ) -> tuple[int, int]:
        """노드 먼저, 엣지 나중에 저장 (엣지가 노드를 참조하므로 순서 중요)."""
        n = await self.save_nodes(nodes)
        e = await self.save_edges(edges)
        return n, e

    async def get_subgraph(self, anchor_id: str, hops: int = 2) -> dict:
        """
        특정 앵커 노드에서 N-hop 서브그래프를 조회한다.

        Returns: {"nodes": [...], "edges": [...]} — 비활성/실패 시 빈 결과.
        """
        if not self.is_active():
            return {"nodes": [], "edges": []}
        hops = max(1, min(int(hops), 5))  # 폭주 방지
        try:
            async with self.driver.session(database=self.database) as session:
                query = (
                    f"MATCH path = (n {{node_id: $anchor}})-[*1..{hops}]-(m) "
                    f"WITH nodes(path) AS ns, relationships(path) AS rs "
                    f"UNWIND ns AS nd "
                    f"WITH collect(DISTINCT nd) AS nodes, rs "
                    f"UNWIND rs AS rel "
                    f"RETURN nodes, collect(DISTINCT rel) AS rels"
                )
                result = await session.run(query, anchor=anchor_id)
                rec = await result.single()
                if not rec:
                    return {"nodes": [], "edges": []}
                nodes = [dict(nd) for nd in rec["nodes"]]
                edges = [
                    {"type": rel.type, **dict(rel)} for rel in rec["rels"]
                ]
                return {"nodes": nodes, "edges": edges}
        except Exception as e:
            logger.warning(f"[GraphStore] 서브그래프 조회 실패: {e}")
            return {"nodes": [], "edges": []}

    async def close(self):
        """드라이버 종료."""
        if self.driver is not None:
            try:
                await self.driver.close()
                logger.info("[GraphStore] Neo4j driver 종료")
            except Exception as e:
                logger.debug(f"[GraphStore] driver 종료 중 무시된 예외: {e}")
            finally:
                self.driver = None
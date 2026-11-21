"""Semantic search ranks exactly and still honours liveness.

Message and conclusion search rank candidates on the embedding table alone and
apply liveness afterwards. These tests pin the observable contract: results are
the exact nearest live rows in distance order, including when the nearest rows
are tombstoned and the search has to fall back to the single-statement query.
"""

import math
from datetime import UTC, datetime

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.config import settings
from src.crud import document as document_crud
from src.crud import message as message_crud
from src.models import Peer, Workspace

DIMS = settings.EMBEDDING.VECTOR_DIMENSIONS
QUERY = [1.0] + [0.0] * (DIMS - 1)


def _at_angle(i: int) -> list[float]:
    """Unit vector whose cosine distance to QUERY grows with i."""
    theta = 0.03 * (i + 1)
    return [math.cos(theta), math.sin(theta)] + [0.0] * (DIMS - 2)


async def _seed_messages(
    db: AsyncSession, workspace: Workspace, peer: Peer, dead_nearest: int
) -> list[str]:
    """30 embedded messages, nearest first; the first `dead_nearest` sit in a tombstoned session."""
    dead = models.Session(workspace_name=workspace.name, name=str(generate_nanoid()))
    live = models.Session(workspace_name=workspace.name, name=str(generate_nanoid()))
    db.add_all([dead, live])
    await db.commit()

    ids: list[str] = []
    for i in range(30):
        session = dead if i < dead_nearest else live
        public_id = str(generate_nanoid())
        ids.append(public_id)
        db.add(
            models.Message(
                public_id=public_id,
                session_name=session.name,
                workspace_name=workspace.name,
                peer_name=peer.name,
                content=f"message {i}",
                seq_in_session=i + 1,
            )
        )
    await db.commit()
    for i, public_id in enumerate(ids):
        db.add(
            models.MessageEmbedding(
                content=f"message {i}",
                message_id=public_id,
                workspace_name=workspace.name,
                session_name=dead.name if i < dead_nearest else live.name,
                peer_name=peer.name,
                sync_state="synced",
                embedding=_at_angle(i),
            )
        )
    await db.execute(
        update(models.Session)
        .where(models.Session.workspace_name == workspace.name)
        .where(models.Session.name == dead.name)
        .values(deleted_at=datetime.now(UTC))
    )
    await db.commit()
    return ids


async def _matched(db: AsyncSession, workspace: Workspace) -> list[str]:
    snippets = await message_crud._search_messages_pgvector(  # pyright: ignore[reportPrivateUsage]
        db, workspace.name, None, query_embedding=QUERY, limit=5, context_window=0
    )
    return [msg.public_id for matched, _context in snippets for msg in matched]


@pytest.mark.asyncio
async def test_message_search_returns_exact_nearest_live_messages(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    ids = await _seed_messages(db_session, workspace, peer, dead_nearest=3)

    assert set(await _matched(db_session, workspace)) == set(ids[3:8])


@pytest.mark.asyncio
async def test_message_search_falls_back_when_candidates_are_all_dead(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, peer = sample_data
    # More tombstoned rows than the candidate window (limit * 4 = 20).
    ids = await _seed_messages(db_session, workspace, peer, dead_nearest=25)

    assert set(await _matched(db_session, workspace)) == set(ids[25:30])


@pytest.mark.asyncio
async def test_conclusion_search_returns_exact_nearest_live_conclusions(
    db_session: AsyncSession, sample_data: tuple[Workspace, Peer]
):
    workspace, observer = sample_data
    observed = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(observed)
    await db_session.flush()
    db_session.add(
        models.Collection(
            workspace_name=workspace.name,
            observer=observer.name,
            observed=observed.name,
        )
    )
    await db_session.flush()

    documents = [
        models.Document(
            workspace_name=workspace.name,
            observer=observer.name,
            observed=observed.name,
            content=f"conclusion {i}",
            embedding=_at_angle(i),
            level="explicit",
            deleted_at=datetime.now(UTC) if i < 2 else None,
        )
        for i in range(30)
    ]
    db_session.add_all(documents)
    await db_session.commit()

    result = await document_crud._query_documents_pgvector(  # pyright: ignore[reportPrivateUsage]
        db_session,
        workspace.name,
        observer.name,
        observed.name,
        QUERY,
        None,
        None,
        5,
    )

    assert [doc.id for doc in result] == [doc.id for doc in documents[2:7]]

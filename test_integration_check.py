"""Quick integration smoke test for the multi-tenant DB refactor."""
import asyncio
import os
import tempfile
import time

TMPDIR = tempfile.gettempdir()


async def main():
    from mnemon.core.persistence import EROSDatabase, TenantConnectionPool, migrate_from_legacy
    from mnemon.core.signal_db import SignalDatabase
    from mnemon.core.models import BondedMemory, MemoryLayer

    # ── Test 1: EROSDatabase per-tenant path ──────────────────────────
    db = EROSDatabase(tenant_id="corp_a", db_dir=TMPDIR)
    assert "mnemon_tenant_corp_a.db" in db.db_path, db.db_path
    await db.connect()
    mem = BondedMemory(
        memory_id="m1",
        tenant_id="corp_a",
        layer=MemoryLayer.EPISODIC,
        content={"text": "hello"},
        timestamp=time.time(),
        last_accessed=time.time(),
    )
    await db.write_memory(mem)
    fetched = await db.fetch_memories("corp_a", ["m1"])
    assert len(fetched) == 1 and fetched[0].content == {"text": "hello"}
    await db.disconnect()
    print("PASS 1: EROSDatabase per-tenant file + write/read")

    # ── Test 2: TenantConnectionPool LRU eviction ─────────────────────
    pool = TenantConnectionPool(db_dir=TMPDIR, max_connections=3)
    await pool.get("alpha")
    await pool.get("beta")
    await pool.get("gamma")
    await pool.get("delta")  # should evict 'alpha' (LRU)
    assert "alpha" not in pool._pool, f"Pool keys: {list(pool._pool.keys())}"
    assert len(pool._pool) == 3
    await pool.close_all()
    print("PASS 2: TenantConnectionPool LRU eviction")

    # ── Test 3: SignalDatabase all methods ────────────────────────────
    sig_path = os.path.join(TMPDIR, "test_signal_int.db")
    sdb = SignalDatabase(sig_path)
    await sdb.connect()
    await sdb.record_fragment_success("abc123abc123abc123abc123abc123", "coding")
    await sdb.record_fragment_success("abc123abc123abc123abc123abc123", "coding")
    await sdb.record_fragment_failure("abc123abc123abc123abc123abc123", "coding")
    sig = await sdb.get_fragment_signal("abc123abc123abc123abc123abc123")
    assert sig["success_count"] == 2, sig
    assert sig["failure_count"] == 1, sig
    assert abs(sig["success_rate"] - 2 / 3) < 0.001, sig
    top = await sdb.get_top_fragments("coding")
    assert len(top) == 1, top
    await sdb.update_vocab_weight("python", 0.3)
    weights = await sdb.get_vocab_weights(["python", "java"])
    assert abs(weights["python"] - 0.8) < 0.001, weights
    assert weights["java"] == 0.5, weights  # unknown → neutral
    await sdb.disconnect()
    print("PASS 3: SignalDatabase all methods")

    # ── Test 4: GDPR delete_tenant ────────────────────────────────────
    pool2 = TenantConnectionPool(db_dir=TMPDIR, max_connections=10)
    await pool2.get("gdpr_test")
    gdpr_path = os.path.join(TMPDIR, "mnemon_tenant_gdpr_test.db")
    assert os.path.exists(gdpr_path), gdpr_path
    await pool2.delete_tenant("gdpr_test")
    assert "gdpr_test" not in pool2._pool
    assert not os.path.exists(gdpr_path), "File should be deleted"
    print("PASS 4: delete_tenant GDPR erasure")

    # ── Test 5: EME warm() with signal_db ────────────────────────────
    from mnemon.core.eme import ExecutionMemoryEngine
    db2 = EROSDatabase(tenant_id="eme_test", db_dir=TMPDIR)
    await db2.connect()
    sdb2 = SignalDatabase(os.path.join(TMPDIR, "sig_eme.db"))
    await sdb2.connect()
    eme = ExecutionMemoryEngine(tenant_id="eme_test", db=db2, signal_db=sdb2)
    await eme.warm()
    await db2.disconnect()
    await sdb2.disconnect()
    print("PASS 5: EME warm() with signal_db (no crash)")

    # ── Test 6: CognitiveMemorySystem accepts signal_db ───────────────
    from mnemon.core.memory import CognitiveMemorySystem
    from mnemon.core.persistence import InvertedIndex
    db3 = EROSDatabase(tenant_id="mem_test", db_dir=TMPDIR)
    await db3.connect()
    sdb3 = SignalDatabase(os.path.join(TMPDIR, "sig_mem.db"))
    await sdb3.connect()
    idx = InvertedIndex()
    cms = CognitiveMemorySystem(
        tenant_id="mem_test", db=db3, index=idx, signal_db=sdb3
    )
    assert cms.signal_db is sdb3
    await db3.disconnect()
    await sdb3.disconnect()
    print("PASS 6: CognitiveMemorySystem accepts signal_db")

    # ── Test 7: migrate_from_legacy is callable ───────────────────────
    assert callable(migrate_from_legacy)
    print("PASS 7: migrate_from_legacy importable")

    # ── Test 8: system_db has no persistence imports ──────────────────
    import mnemon.core.system_db as sysdb
    src = open("mnemon/core/system_db.py").read()
    assert "from .persistence" not in src and "import EROSDatabase" not in src
    assert "mnemon_system.db" in src
    print("PASS 8: system_db.py fully isolated")

    # ── Cleanup temp files ────────────────────────────────────────────
    for fname in [
        "mnemon_tenant_corp_a.db", "mnemon_tenant_alpha.db",
        "mnemon_tenant_beta.db", "mnemon_tenant_gamma.db",
        "mnemon_tenant_delta.db", "mnemon_tenant_eme_test.db",
        "mnemon_tenant_mem_test.db",
        "test_signal_int.db", "sig_eme.db", "sig_mem.db",
    ]:
        p = os.path.join(TMPDIR, fname)
        if os.path.exists(p):
            os.remove(p)

    print()
    print("All 8 integration tests PASSED.")


asyncio.run(main())

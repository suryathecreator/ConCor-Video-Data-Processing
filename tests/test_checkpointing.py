from __future__ import annotations

import threading
from concor_video.checkpointing import sample_claim


def test_concurrent_claim_has_exactly_one_owner(tmp_path) -> None:
    claim = tmp_path / "video.claim"
    entered = threading.Barrier(2)
    release = threading.Event()
    results = []
    lock = threading.Lock()

    def contender(owner: str) -> None:
        entered.wait()
        with sample_claim(claim, owner=owner, heartbeat_seconds=1) as acquired:
            with lock:
                results.append(acquired)
            if acquired:
                release.wait(timeout=3)

    threads = [threading.Thread(target=contender, args=(f"worker-{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    while len(results) < 2:
        pass
    release.set()
    for thread in threads:
        thread.join(timeout=5)
    assert sorted(results) == [False, True]
    assert not claim.exists()

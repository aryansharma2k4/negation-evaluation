"""Phase 4.1: write data/paragraph_eval.jsonl (machine-authored) and the human stub.

    ./venv/bin/python experiments/f1/make_paragraphs.py

Every item was written by the model (Claude) for report_f1.md, not by a teacher or student, and is
labelled ``authorship: machine`` in the file. Gold per teacher sentence: covered / contradicted /
missing, with the index of the student sentence that covers or contradicts it; gold extras = student
sentences that match no teacher sentence. Student sentences are deliberately not in teacher order.
"""

from __future__ import annotations

import json

from core import REPO

C, X, M = "covered", "contradicted", "missing"

# (id, subject, tags, teacher sentences, student sentences, [(verdict, student idx) per teacher], extras)
ITEMS = [
    ("p01", "OS", ["reworded"],
     ["A process is a program in execution.", "Each process has its own address space.",
      "The operating system keeps a process control block for every process.",
      "A context switch saves the state of the running process."],
     ["When the CPU switches between processes, the state of the current process is saved.",
      "A program that is currently executing is called a process.",
      "For each process the OS maintains a process control block.", "Every process gets a separate address space."],
     [(C, 1), (C, 3), (C, 2), (C, 0)], []),
    ("p02", "OS", ["one_contradiction"],
     ["Round-robin scheduling gives each process a fixed time quantum.", "Round-robin scheduling is preemptive.",
      "A very large time quantum makes round-robin behave like first-come first-served."],
     ["With a huge quantum, round robin degenerates into FCFS.", "Round robin is a non-preemptive scheduling algorithm.",
      "Every process receives the same fixed slice of CPU time under round robin."],
     [(C, 2), (X, 1), (C, 0)], []),
    ("p03", "OS", ["missing"],
     ["A deadlock requires mutual exclusion, hold and wait, no preemption and circular wait.",
      "Deadlock can be prevented by breaking any one of these four conditions.",
      "The banker's algorithm avoids deadlock by checking for safe states.",
      "Deadlock detection lets deadlocks happen and then recovers from them."],
     ["The banker's algorithm keeps the system in a safe state so that deadlock is avoided.",
      "Four conditions, mutual exclusion, hold and wait, no preemption and circular wait, must all hold for deadlock."],
     [(C, 1), (M, None), (C, 0), (M, None)], []),
    ("p04", "OS", ["one_contradiction", "negative_teacher"],
     ["A user-level thread is managed without kernel support.", "The kernel does not know about user-level threads.",
      "If one user-level thread blocks, the whole process can block."],
     ["The kernel is aware of every user-level thread.", "User threads are handled by a library rather than by the kernel.",
      "A blocking call in one user thread may block the entire process."],
     [(C, 1), (X, 0), (C, 2)], []),
    ("p05", "OS", ["reworded", "cancellation"],
     ["Page faults are common when a process starts.", "Demand paging loads a page only when it is referenced.",
      "A page fault is handled by the operating system."],
     ["It is not uncommon for a process to page-fault heavily when it starts.",
      "Pages are brought into memory only when they are first accessed.", "The OS services every page fault."],
     [(C, 0), (C, 1), (C, 2)], []),
    ("p06", "OS", ["extra"],
     ["Virtual memory lets a process use more memory than is physically installed.",
      "The page table maps virtual pages to physical frames.", "The TLB caches recent page table entries."],
     ["Using virtual memory, a program can be larger than physical RAM.",
      "Virtual to physical translation is done through the page table.",
      "The translation lookaside buffer stores recently used translations.",
      "Linux is an operating system that supports virtual memory."],
     [(C, 0), (C, 1), (C, 2)], [3]),
    ("p07", "OS", ["reworded", "negative_teacher"],
     ["A semaphore is an integer variable used for synchronization.", "The wait operation decrements the semaphore.",
      "A binary semaphore cannot take values other than 0 and 1."],
     ["A binary semaphore only ever holds the values 0 or 1.", "Semaphores are integer variables used to synchronize processes.",
      "Calling wait on a semaphore decreases its value."],
     [(C, 1), (C, 2), (C, 0)], []),
    ("p08", "OS", ["one_contradiction", "extra"],
     ["The FCFS scheduling algorithm is simple to implement.", "FCFS can cause the convoy effect.",
      "Short processes may wait a long time behind a long process under FCFS."],
     ["First-come first-served is difficult to implement.", "FCFS suffers from the convoy effect.",
      "Under FCFS a short job can wait behind a long one for a long time.",
      "SJF gives the minimum average waiting time."],
     [(X, 0), (C, 1), (C, 2)], [3]),
    ("p09", "OS", ["missing", "extra"],
     ["Thrashing occurs when a process spends more time paging than executing.",
      "Thrashing is caused by too few frames per process.", "The working-set model can be used to prevent thrashing."],
     ["When a process lacks enough frames, it thrashes.", "Paging is a memory management scheme.",
      "LRU replaces the page that was used least recently."],
     [(M, None), (C, 0), (M, None)], [1, 2]),
    ("p10", "CN", ["reworded"],
     ["TCP is a connection-oriented protocol.", "TCP guarantees reliable, in-order delivery of bytes.",
      "TCP uses a three-way handshake to set up a connection.", "TCP performs flow control with a sliding window."],
     ["A connection must be established before TCP sends data.",
      "Connection setup in TCP uses three messages: SYN, SYN-ACK and ACK.",
      "Bytes sent over TCP arrive reliably and in order.", "A sliding window lets TCP control the flow of data."],
     [(C, 0), (C, 2), (C, 1), (C, 3)], []),
    ("p11", "CN", ["one_contradiction", "negative_teacher"],
     ["UDP is a connectionless protocol.", "UDP does not guarantee delivery.", "UDP has lower overhead than TCP."],
     ["UDP guarantees that every datagram is delivered.", "No connection is set up before UDP sends a datagram.",
      "UDP headers are smaller, so UDP has less overhead than TCP."],
     [(C, 1), (X, 0), (C, 2)], []),
    ("p12", "CN", ["reworded", "cancellation"],
     ["IP is a best-effort protocol.", "Packets sent with IP may be lost or duplicated.", "IP routes each packet independently."],
     ["It is not impossible for an IP packet to be lost or duplicated.",
      "IP makes no guarantees and only tries its best to deliver packets.", "Each IP packet is routed on its own."],
     [(C, 1), (C, 0), (C, 2)], []),
    ("p13", "CN", ["one_contradiction"],
     ["Every host on the internet needs an IP address.", "DNS translates domain names into IP addresses.",
      "DNS uses a hierarchical, distributed database."],
     ["Domain names are converted to IP addresses by DNS.", "DNS data is stored in one central database.",
      "An internet host must have an IP address."],
     [(C, 2), (C, 0), (X, 1)], []),
    ("p14", "CN", ["missing"],
     ["The data link layer frames packets for transmission.", "Ethernet uses MAC addresses to identify devices.",
      "A switch forwards frames based on MAC addresses.", "CSMA/CD detects collisions on a shared medium."],
     ["Switches use MAC addresses to decide where to forward frames.", "Devices on Ethernet are identified by MAC addresses."],
     [(M, None), (C, 1), (C, 0), (M, None)], []),
    ("p15", "CN", ["extra"],
     ["Routers forward packets between networks.", "Routing tables tell a router where to send a packet.",
      "Distance-vector routing shares tables with neighbours."],
     ["A router moves packets from one network to another.", "Each router looks up the destination in its routing table.",
      "In distance-vector routing, routers exchange their tables with neighbouring routers.",
      "OSPF is a link-state protocol.", "BGP is used between autonomous systems."],
     [(C, 0), (C, 1), (C, 2)], [3, 4]),
    ("p16", "CN", ["one_contradiction", "negative_teacher"],
     ["Congestion control prevents the network from being overloaded.",
      "TCP reduces its congestion window when it detects packet loss.", "Slow start does not begin with a large window."],
     ["Slow start begins with a small congestion window.", "When TCP detects loss it increases its congestion window.",
      "The goal of congestion control is to avoid overloading the network."],
     [(C, 2), (X, 1), (C, 0)], []),
    ("p17", "CN", ["reworded", "cancellation"],
     ["A checksum detects errors in transmitted data.", "CRC is stronger than a simple checksum.",
      "Error detection codes add redundant bits to the data."],
     ["Redundant bits are added to data so that errors can be detected.",
      "It is not untrue that CRC is more powerful than a plain checksum.", "Transmission errors can be found using a checksum."],
     [(C, 2), (C, 1), (C, 0)], []),
    ("p18", "CN", ["one_contradiction", "conditional"],
     ["If the receive window is full, the sender must stop sending.",
      "The receiver advertises its window size in every ACK.", "Flow control protects a slow receiver."],
     ["Each acknowledgement carries the receiver's advertised window.",
      "A slow receiver is protected from being overwhelmed by flow control.",
      "Even when the receive window is full, the sender keeps sending."],
     [(X, 2), (C, 0), (C, 1)], []),
    ("p19", "DBMS", ["reworded", "negative_teacher"],
     ["A primary key uniquely identifies each row in a table.", "A primary key cannot contain null values.",
      "A foreign key refers to the primary key of another table."],
     ["Each row of a table is identified uniquely by its primary key.", "Null values are not allowed in a primary key column.",
      "A foreign key points to the primary key of a different table."],
     [(C, 0), (C, 1), (C, 2)], []),
    ("p20", "DBMS", ["one_contradiction", "negative_teacher"],
     ["A transaction is a unit of work that is atomic.", "Atomicity means a transaction is never partially applied.",
      "Durability means committed changes survive a crash."],
     ["Once a transaction commits, its changes persist even after a crash.", "A transaction is an atomic unit of work.",
      "Atomicity means a transaction is sometimes partially applied."],
     [(C, 1), (X, 2), (C, 0)], []),
    ("p21", "DBMS", ["missing"],
     ["Normalization reduces redundancy in a database.", "A table is in first normal form if every attribute is atomic.",
      "Second normal form removes partial dependencies.", "Third normal form removes transitive dependencies."],
     ["Normalizing a database cuts down on redundant data.", "In 3NF there are no transitive dependencies."],
     [(C, 0), (M, None), (M, None), (C, 1)], []),
    ("p22", "DBMS", ["extra"],
     ["An index speeds up data retrieval.", "Indexes slow down inserts and updates.", "A B+ tree is a common index structure."],
     ["Retrieval becomes faster with an index.", "Many databases use B+ trees for their indexes.",
      "Writes become slower because indexes must be updated.", "SQL is a declarative query language."],
     [(C, 0), (C, 2), (C, 1)], [3]),
    ("p23", "DBMS", ["one_contradiction"],
     ["SQL is a declarative language.", "A SELECT statement retrieves rows from a table.", "The WHERE clause filters rows."],
     ["SQL is a procedural language.", "Rows are filtered using the WHERE clause.", "Rows of a table are fetched with SELECT."],
     [(X, 0), (C, 2), (C, 1)], []),
    ("p24", "DBMS", ["reworded", "negative_teacher", "cancellation"],
     ["Deadlocks between transactions are not rare in busy databases.",
      "Two-phase locking guarantees conflict serializability.", "Strict two-phase locking holds exclusive locks until commit."],
     ["Deadlocks between transactions are common in busy databases.",
      "Conflict-serializable schedules are guaranteed by two-phase locking.",
      "Under strict 2PL, exclusive locks are released only after the transaction commits."],
     [(C, 0), (C, 1), (C, 2)], []),
    ("p25", "DBMS", ["one_contradiction", "negative_teacher"],
     ["Every relation in a relational database has a schema.", "A view is a virtual table defined by a query.",
      "Views do not store data themselves."],
     ["A view is a virtual table built from a query.", "Views store their own copy of the data.", "Each relation has a schema."],
     [(C, 2), (C, 0), (X, 1)], []),
    ("p26", "DBMS", ["missing", "extra"],
     ["A database management system stores and manages data.", "A DBMS enforces integrity constraints.",
      "A DBMS provides concurrency control for multiple users.", "A DBMS supports recovery after failures."],
     ["A DBMS lets many users access data at the same time safely.",
      "The DBMS makes sure integrity constraints are not violated.", "MySQL and PostgreSQL are popular database systems."],
     [(M, None), (C, 1), (C, 0), (M, None)], [2]),
    ("p27", "OS", ["one_contradiction", "scope"],
     ["All processes in the ready queue are waiting for the CPU.",
      "The long-term scheduler controls the degree of multiprogramming.", "The short-term scheduler runs very frequently."],
     ["How many processes are in memory is decided by the long-term scheduler.",
      "Not all processes in the ready queue are waiting for the CPU.", "The CPU scheduler is invoked very often."],
     [(X, 1), (C, 0), (C, 2)], []),
    ("p28", "CN", ["reworded"],
     ["The OSI model has seven layers.", "The physical layer transmits raw bits.", "The network layer handles routing.",
      "The transport layer provides end-to-end communication.", "The application layer is closest to the user.",
      "Each layer provides services to the layer above it."],
     ["Services flow upward: each layer serves the one above.", "Routing is the job of the network layer.",
      "Seven layers make up the OSI reference model.", "Raw bits are sent over the medium by the physical layer.",
      "End-to-end delivery between processes is provided by the transport layer.",
      "The layer nearest to the user is the application layer."],
     [(C, 2), (C, 3), (C, 1), (C, 4), (C, 5), (C, 0)], []),
]

SCHEMA = ("one JSON object per line: id, subject (OS / CN / DBMS), authorship (machine | human), tags, "
          "teacher [sentences], student [sentences], gold_teacher [{verdict: covered | contradicted | missing, "
          "student: index or null}], gold_extra [student indices matched to no teacher sentence]")


def main() -> None:
    rows = []
    for pid, subj, tags, teacher, student, gold, extra in ITEMS:
        assert 3 <= len(teacher) <= 6 and 2 <= len(student) <= 6, pid
        assert len(gold) == len(teacher), pid
        used = [s for v, s in gold if s is not None]
        assert len(used) == len(set(used)), f"{pid}: a student sentence is matched twice"
        assert all((v == M) == (s is None) for v, s in gold), pid
        assert sorted(used + extra) == list(range(len(student))), f"{pid}: every student sentence matched or extra"
        rows.append({"id": pid, "subject": subj, "authorship": "machine",
                     "author_note": "written by the model (Claude) for report_f1.md step 4.1",
                     "tags": tags, "teacher": teacher, "student": student,
                     "gold_teacher": [{"verdict": v, "student": s} for v, s in gold], "gold_extra": extra})
    with (REPO / "data" / "paragraph_eval.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    human = REPO / "data" / "paragraph_eval_human.jsonl"
    if not human.exists():
        human.write_text("")
    from collections import Counter
    print(len(rows), "items;", Counter(t for r in rows for t in r["tags"]),
          sum(len(r["teacher"]) for r in rows), "teacher sentences")


if __name__ == "__main__":
    main()

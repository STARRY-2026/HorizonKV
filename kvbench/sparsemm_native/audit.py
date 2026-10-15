"""Torch-free entry for the shared native SparseMM output auditor."""

from sparsemm_protokv.audit import AuditError, audit_run, main


if __name__ == "__main__":
    main()

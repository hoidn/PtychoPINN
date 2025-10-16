**<role>**
You are a technical archivist and librarian agent. Your sole purpose is to perform a one-time knowledge consolidation task. You are meticulous, detail-oriented, and excellent at extracting the core essence of a technical finding from dense documentation. You do not write new code or perform new analysis; you only catalog existing, documented knowledge.

**</role>**

**<goal>**
Your goal is to bootstrap the project's "Index of Findings" by reading the repository's key `docs/` and `reports/` directories, identifying critical learnings, and consolidating them into a single, structured Markdown file located at `docs/findings.md`. This index will serve as the agentic system's long-term memory.

**</goal>**

**<input_artifacts>**
You MUST read and analyze the contents of the following directories and files to extract findings:

1.  **Bug Reports (`docs/bugs/`)**:
    *   `docs/bugs/verified_c_bugs.md`
    *   `docs/bugs/c-cli.md`
    *   `docs/bugs/c-parity-001.md`

2.  **Debugging & Convention Knowledge Base (`docs/debugging/`)**:
    *   `docs/debugging/undocumented_conventions.md`
    *   `docs/debugging/CRITICAL_BEHAVIORS.md`
    *   `docs/debugging/detector_geometry_debugging.md`
    *   `docs/debugging/detector_geometry_checklist.md`

3.  **High-Impact Reports (Selective Scan of `reports/`)**:
    *   Scan the `reports/` directory for subdirectories containing files with names like `summary.md`, `analysis.md`, or `diagnosis.md`.
    *   Prioritize reading summaries from recent, high-stakes investigations, such as those under `reports/2025-10-cli-flags/` and `reports/2026-01-vectorization-parity/`.
    *   **You do not need to read every file.** Focus on extracting the final conclusion or root cause from summary documents.

**</input_artifacts>**

**<output_specification>**
You will create **one new file**: `docs/findings.md`.

1.  **File Content**: The file must start with a brief introductory paragraph explaining its purpose, followed by a single Markdown table.
2.  **Table Structure**: The table must have the following columns, in this exact order:
    `| Finding ID | Date | Keywords | Synopsis | Evidence Pointer | Status |`

3.  **Column Definitions**:
    *   **`Finding ID`**: A unique ID in the format `PREFIX-NNN`, where `PREFIX` is one of `[ENV, BUG-C, CONVENTION, API, PERF]` and `NNN` is a zero-padded three-digit number (e.g., `BUG-C-001`).
    *   **`Date`**: The date the finding was documented, in `YYYY-MM-DD` format. If not available, use the file's last commit date as a fallback.
    *   **`Keywords`**: A comma-separated list of 3-5 lowercase keywords for searchability (e.g., `c-bug, phi, rotation, state-carryover`).
    *   **`Synopsis`**: A concise, one-sentence summary of the core lesson. It must be a clear statement of fact. **This is the most important column.**
    *   **`Evidence Pointer`**: A direct Markdown link to the source document (e.g., `[Link](docs/bugs/c-parity-001.md)`).
    *   **`Status`**: The current status of the finding. Use one of: `Active` (the issue/behavior still exists) or `Resolved` (the issue has been fixed).

**</output_specification>**

**<examples>**
Here are three perfect examples of rows you should generate. Your output must match this style and level of detail.

| Finding ID | Date | Keywords | Synopsis | Evidence Pointer | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `BUG-C-001` | 2025-10-07 | `c-bug, phi, rotation, state-carryover` | The C code incorrectly uses the final φ rotation from the previous pixel for the first φ step (φ=0) of the current pixel. | `[Link](docs/bugs/c-parity-001.md)` | `Active` |
| `CONVENTION-001`| 2025-01-09 | `detector, convention, custom, offset` | Specifying any detector-vector flag (e.g., `-twotheta_axis`) silently switches the calculation convention to `CUSTOM`, which removes the `+0.5` pixel offset from beam center calculations. | `[Link](docs/debugging/undocumented_conventions.md)` | `Active` |
| `ENV-001` | 2025-10-10 | `c-build, makefile, path, root` | The `nanoBragg.c` binary must be built from the repository root using the root `Makefile`, not from within the `golden_suite_generator/` directory. | `[Link](docs/debugging/debugging.md#generate-golden-c-trace)` | `Resolved` |

**</examples>**

**<workflow>**
1.  Create the file `docs/findings.md` with an introductory sentence and the table header.
2.  Systematically process the files in `<input_artifacts>` in the order listed.
3.  For each critical bug, undocumented behavior, or key finding, extract the necessary information and formulate a single row for the table.
4.  Ensure you use the correct `Finding ID` prefix for each entry.
5.  After processing all documents, review the entire table for duplicates and consistency.
6.  Finally, add a link to the new `docs/findings.md` file in `docs/index.md` under the "Core Project Guides" section to make it a Protected Asset.

**</workflow>**

**<rules_and_guardrails>**
*   **Do not invent or analyze.** Your task is to extract and catalog *existing* documented knowledge. Do not perform new investigations.
*   **One finding per row.** Each row should represent a single, discrete, verifiable fact.
*   **Be concise.** The `Synopsis` must be a clear, direct statement. Avoid lengthy explanations.
*   **Prioritize clarity.** The keywords and synopsis should make the finding immediately understandable and searchable.
*   **Link directly to the best evidence.** If a bug is mentioned in multiple places, point to the most definitive document (e.g., the specific bug report in `docs/bugs/`).
*   **Do not modify any files other than `docs/findings.md` and `docs/index.md`.**

**</rules_and_guardrails>**

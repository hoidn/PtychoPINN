<role>Software architect</role>

## Context

## High-Level Objective

Identify all files that would need modifications to implement a set of changes

## Mid-Level Objective

- Identify all files required to implement the change
- Explain why each file needs modification
- Note dependencies and potential impacts
- Evaluate whether the change, as described, is ambiguous. If so, generate questions to clarify

## Implementation Notes

- Don't speculatively suggest any changes unless the change is unambiguous, given the <changes> description

### Beginning Context

- `context` (concatenation of all potentially relevant files to review)

### Ending Context

- `tochange.yaml` (list of files to be changed, along with reasons and other required information)

```aider
You will review the provided system to identify all files that would need modifications to implement the following change: 
    <changes> description: '<description>'
```

<task>
1. Analyze the documentation and / or codebase to identify all files that directly or indirectly inform or depend on any aspects of the <changes> description task

2. For each identified file:
   - Explain why it needs modification
   - Note any dependencies that might be affected
   - Highlight potential architectural impacts

3. Group the files by:
   - Core functionality
   - Secondary functionality
   - Documentation
</task>

<output_format>
create a new file called tochange.yaml with the following:
Please structure your response as:

# Files Requiring Updates
[For each file:]
## [filename].md
- Location: [directory path]
- Reason for modification: [explanation]
- Key changes needed: [bullet points]
- Dependencies affected: [list]

# Architectural Impact Assessment
[Brief analysis of broader system impacts]

# Implementation Order Recommendation
[Suggested sequence for making the changes]

# Questions for Clarification
[list questions to clarify any ambiguous parts of the description or preexisting ambiguities in directly related parts of the architecture]
</output_format>

<thought_process>
1. First identify direct references to the topic in the description
2. Then trace dependencies through system interfaces
3. Map out affected codebase and documentation sections
</thought_process>

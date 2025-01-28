<role>Software architect</role>
## Context

## High-Level Objective

Write a spec prompt according to the given description and Q&A clarifications. 

## Mid-Level Objective

- Explain why each file needs modification
- Note dependencies and potential impacts
- Evaluate whether the change, as described, is ambiguous. Review the q and a to see whether the ambiguity is addressed.
- Draft the necessary structural changes and behavioral changes in the format of a spec prompt, as documented in the spec prompt guide

### Beginning Context

- tochange.yaml, containing the files that need to be modified and other relevant information
- Description of TODO / <high-level objective>: see <desc> 
- Questions and answers: see <quest>
- Spec prompt guide: see <guide>

### Ending Context

- taskspec.md, containing a well-formed spec prompt documenting the changes necessary to implement <high-level objective>

<output_format>
1. First, summarize understanding:
   - Key points from the Q&A discussion
   - Implications for the implementation
   - How this affects the requested changes

2. Then draft a well-formed spec prompt instructing the changes necessary to implement the <high-level objective>. It should specify which components need to be added, removed, modified, and what the behavioral changes are, but it should not include implementation details. This spec prompt should be enclosed in <taskspec> tags.

</output_format>

<desc>
'<description>'
</desc>

<quest>
'<questions>'
</quest>

<guide>
'<spec_prompt_guide>'
</guide>

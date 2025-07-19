Run this shell command:

`gemini -p "<user query> $ARGUMENTS </user query> <task> The user wants context information about the <user query>:  First, review the project documentation, starting from core docs such as CLAUDE.md and then expanding to the others. Then, review each file (including documentation, code and configuration) and for each, think about whether or not that file is relevant to the user task. Second, for each relevant file that you identified in the previous step, think about why that file is relevant and quote any key sections. Return your answer in the form of <return format> a list of files with a justification for each included file and with quotes of key sections that you identified </return format>.</task> "`

Do not do any analysis until AFTER reading the output of the shell command.
The command will return file paths and additional text, per <return format>. Given the <user query>, read each of those files. 
Then, return a literal quote of the shell command output, followed by your own analysis of the <user query>. 


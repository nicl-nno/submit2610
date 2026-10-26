# Anonymization Notes

This artifact was prepared for double-anonymous review.

The following material was removed or normalized:

- author names, personal account handles, e-mail addresses, and affiliations;
- private hostnames, addresses, usernames, and cluster-specific absolute paths;
- internal scheduler scripts and working notes;
- manuscript and slide sources that are not needed to execute the code;
- identity-specific probe output and large teacher transcripts;
- machine-specific directory names embedded in benchmark outputs.

Research questions, answers, task prompts, and test fixtures were retained in
their original language. Where those records contained author names,
biographical details, affiliations, personal accounts, or private
infrastructure, only the identifying spans were replaced with explicit
anonymous placeholders. Records were not replaced with synthetic English
examples.

Russian-language documentation and comments were removed from executable and configuration files. Russian text was retained in prompts, datasets, expected outputs, regular expressions, linguistic resources, and tests when removing it would change the task or invalidate the experiments.

Neutral values such as `anonymous/base-model`, `anonymous@example.com`, `cluster.example.invalid`, and `/path/to/storage` are placeholders. They do not identify live resources and must be replaced with local values when running the artifact.

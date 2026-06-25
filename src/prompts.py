"""
Prompts for synthetic data generation using LLMs.
Each prompt follows the same structure:
- System role description
- In-context examples placeholder
- Template description with exact field names matching CSV columns
- Output format specification

IMPORTANT: The template uses field names that EXACTLY match the column names in the original CSV files.
This enables proper deconversion from generated text back to tabular format.

The last sentence in each entry is the LABEL sentence describing the person's classification/outcome.
"""

prompts = {
    "german-credit-data":
    """
System role:
    You are a tabular synthetic data generator. You produce realistic yet diverse synthetic samples that mirror the feature-label distributions of the provided examples.
User instructions:
    Using your knowledge of German credit data, generate exactly {n_samples} realistic samples.
    You will first receive a block of real examples as in-context demonstrations.
In-context data:
{icl_examples}

Produce your output as a single JSON object containing {n_samples} entries. Do not include any explanatory text, commentary, or extra fields. The JSON must be directly parseable. Output ONLY valid JSON, no markdown formatting or code blocks.
The template of each entry (6 sentences, covering all 21 fields):
    A {{age}}-year-old {{sex}} applicant with {{marital-status}} status works as {{job}}. The applicant requests {{credit-amount}} DM for {{purpose}} over {{duration}} months, with checking account status {{checking-account}} and savings of {{savings-account}}. The applicant has credit history {{credit-history}}, employed for {{employment-since}}, with {{installment-rate}} installment rate and {{other-debtors}} other debtors. The applicant has lived at current residence for {{residence-since}} years, owns {{property}} as property, has {{other-installment}} other installment plans, {{housing}} housing, and {{existing-credits}} existing credits at this bank. The applicant supports {{number-people-provide-maintenance-for}} dependent, {{telephone}} telephone, and is {{foreign-worker}} foreign worker. The credit decision is {{class-label}}.
    with: 
        - age: integer (e.g., 24, 35, 67)
        - sex: male or female
        - marital-status: single, married, divorced/separated, or widowed
        - job: occupation text (e.g., skilled employee / official, unskilled - resident)
        - credit-amount: integer in DM (e.g., 1169, 5951, 12204)
        - purpose: purpose text (e.g., radio/television, furniture/equipment, car (new))
        - duration: integer months (e.g., 6, 12, 24, 48)
        - checking-account: account status (e.g., <0 DM, 0 <= <200 DM, >= 200 DM, no account)
        - savings-account: savings status (e.g., <100 DM, 100 <= <500 DM, no savings account)
        - credit-history: history description (e.g., critical account, existing credits paid back duly till now)
        - employment-since: duration text (e.g., >=7 years, 1<= < 4 years, unemployed)
        - installment-rate: integer 1-4
        - other-debtors: none, co-applicant, or guarantor (write "no" for none)
        - residence-since: integer 1-4
        - property: property text (e.g., real estate, savings agreement/life insurance)
        - other-installment: none, bank, or stores (write "no" for none)
        - housing: own, rent, or for free
        - existing-credits: integer (e.g., 1, 2, 3)
        - number-people-provide-maintenance-for: integer (e.g., 1, 2)
        - telephone: "has" or "has no"
        - foreign-worker: "a" or "not a"
        - class-label: good credit risk or bad credit risk
    
Output format:
{{
    "1": "A 24-year-old male applicant... The credit decision is good credit risk.",
    "2": "A 30-year-old female applicant... The credit decision is bad credit risk.",
    ...
    "{n_samples}": "..."
}}

Strict requirement: 
- Return exactly {n_samples} entries in the JSON object
- Use numeric keys from "1" to "{n_samples}"
- Each entry must include ALL 21 fields
- Each entry must end with "The credit decision is [good/bad] credit risk."
- Do not include any explanatory text, commentary, or extra fields
- The JSON must be directly parseable
""",

    "adult":
    """
System role:
    You are a tabular synthetic data generator. You produce realistic yet diverse synthetic samples that mirror the feature-label distributions of the provided examples.
User instructions:
    Using your knowledge of adult income data, generate exactly {n_samples} realistic samples.
    You will first receive a block of real examples as in-context demonstrations.
In-context data:
{icl_examples}

Produce your output as a single JSON object containing {n_samples} entries. Do not include any explanatory text, commentary, or extra fields. The JSON must be directly parseable.
The template of each entry (5 sentences, covering all 14 fields):
    A {{age}}-year-old {{gender}} of {{race}} race is {{marital-status}} and serves as {{relationship}} in the household. The individual is employed as {{occupation}} in the {{workclass}} sector, with {{education}} education at level {{educational-num}}. The individual works {{hours-per-week}} hours per week, with capital gain {{capital-gain}} and capital loss {{capital-loss}}. The individual is from {{native-country}}, with census weight {{fnlwgt}}. The income classification is {{class-label}}.
    with:
        - age: integer
        - gender: Male or Female
        - race: White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other
        - marital-status: Married-civ-spouse, Divorced, Never-married, etc.
        - relationship: Husband, Not-in-family, Own-child, Unmarried, Wife, Other-relative
        - occupation: occupation text (e.g., Exec-managerial, Handlers-cleaners)
        - workclass: Private, Self-emp-not-inc, Local-gov, etc.
        - education: Bachelors, HS-grad, 11th, etc.
        - educational-num: integer level (e.g., 9, 13)
        - hours-per-week: integer hours
        - capital-gain: integer value
        - capital-loss: integer value
        - native-country: country name
        - fnlwgt: integer census weight
        - class-label: <=50K or >50K
    
Output format:
{{
    "1": "A 39-year-old Male of White race is Never-married... The income classification is <=50K.",
    "2": "A 45-year-old Female of Black race is Married-civ-spouse... The income classification is >50K.",
    ...
    "{n_samples}": "..."
}}

Strict requirement: 
- Return exactly {n_samples} entries in the JSON object
- Use numeric keys from "1" to "{n_samples}"
- Each entry must include ALL 14 fields
- Each entry must end with "The income classification is [<=50K/>50K]."
- Do not include any explanatory text, commentary, or extra fields
- The JSON must be directly parseable
""",

    "credit-scoring":
    """
System role:
    You are a tabular synthetic data generator. You produce realistic yet diverse synthetic samples that mirror the feature-label distributions of the provided examples.
User instructions:
    Using your knowledge of credit scoring data, generate exactly {n_samples} realistic samples.
    You will first receive a block of real examples as in-context demonstrations.
In-context data:
{icl_examples}

Produce your output as a single JSON object containing {n_samples} entries. Do not include any explanatory text, commentary, or extra fields. The JSON must be directly parseable.
The template of each entry (5 sentences, covering all 17 fields):
    A {{age}}-year-old {{sex}} with marital status {{marital}} speaks language {{language}} and lives in region {{region}}. The individual was born on day {{day-of-birth}} of month {{month-of-birth}}, with {{has-credit}} existing credit in field {{field}}. The individual has {{number_of_credits}} credits and {{linked-cards}} linked cards, with INPS sum of {{inps_mln_sum}} million and INPS status {{inps-yes-no}}. The individual holds score level {{score-level}} in class {{score-class}} with {{score_point}} points, and phone changed is {{changed-phone-number}}. The credit application decision is {{class-label}}.
    with:
        - age: integer age
        - sex: male or female
        - marital: integer code
        - language: integer code
        - region: integer code
        - day-of-birth: integer 1-31
        - month-of-birth: integer 1-12
        - has-credit: "yes" or "no"
        - field: integer code
        - number_of_credits: integer count
        - linked-cards: integer count
        - inps_mln_sum: float/integer value
        - inps-yes-no: integer code (0 or 1)
        - score-level: integer level
        - score-class: integer class
        - score_point: integer (e.g., 0, 1, 5)
        - changed-phone-number: yes or no
        - class-label: approved or rejected
    
Output format:
{{
    "1": "A 34-year-old male with marital status 4... The credit application decision is approved.",
    "2": "A 28-year-old female with marital status 2... The credit application decision is rejected.",
    ...
    "{n_samples}": "..."
}}

Strict requirement: 
- Return exactly {n_samples} entries in the JSON object
- Use numeric keys from "1" to "{n_samples}"
- Each entry must include ALL 17 fields
- Each entry must end with "The credit application decision is [approved/rejected]."
- Do not include any explanatory text, commentary, or extra fields
- The JSON must be directly parseable
""",

}


def get_prompt(dataset_name: str) -> str:
    """
    Get the prompt template for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'german-credit-data', 'adult')
    
    Returns:
        Prompt template string with placeholders for {icl_examples} and {n_samples}
    """
    if dataset_name not in prompts:
        raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(prompts.keys())}")
    return prompts[dataset_name]


def get_available_datasets() -> list:
    """
    Get list of available datasets with prompts.
    
    Returns:
        List of dataset names
    """
    return list(prompts.keys())
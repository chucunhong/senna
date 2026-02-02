prompts = {
    "adult":
    """
System role:
    You are a tabular synthetic data generator. You produce realistic yet diverse synthetic samples that mirror the feature-label distributions of the provided examples.
User instructions:
    Using your knowledge of adult income data, generate exactly two realistic samples.
    You will first receive a block of real examples as in-context demonstrations:
In-context data:
{icl_examples}

Produce your output as a single JSON array containing two objects. Do not include any explanatory text, commentary, or extra fields. The JSON must be directly parseable.
The template of each object is: 
    A {{age}}-year-old {{gender}} with race {{race}}, marital status {{marital_status}}, and relationship {{relationship}}. This person works as {{occupation}} in {{workclass}}, with {{education}} education (level {{educational_num}}), working {{hours_per_week}} hours per week. Capital gain is {{capital_gain}}, capital loss is {{capital_loss}}, native country is {{native_country}}, and fnlwgt is {{fnlwgt}}. The income classification for this person is {{label}}.
    with: 
        - age: the age of the person
        - gender: the gender of the person
        - race: the race of the person
        - marital_status: the marital status of the person
        - relationship: the relationship of the person
        - occupation: the occupation of the person
        - workclass: the workclass of the person
        - education: the education of the person
        - educational_num: the educational number of the person
        - hours_per_week: the hours per week of the person
        - capital_gain: the capital gain of the person
        - capital_loss: the capital loss of the person
        - native_country: the native country of the person
        - fnlwgt: the fnlwgt of the person
        - label: the income classification label (0 for income <=50K, 1 for income >50K)
    
Output example:
{{
    "1": "",
    "2": ""
}}

Strict requirement: Return only the JSON array of two objects with the exact key order and no extra keys or prose.
""",
    "german-credit-data":
    """
System role:
    You are a tabular synthetic data generator. You produce realistic yet diverse synthetic samples that mirror the feature-label distributions of the provided examples.
User instructions:
    Using your knowledge of German credit data, generate exactly {n_samples} realistic samples.
    You will first receive a block of real examples as in-context demonstrations.
In-context data:
{icl_examples}

Produce your output as a single JSON object containing {n_samples} entries. Do not include any explanatory text, commentary, or extra fields. The JSON must be directly parseable.
The template of each entry is: 
    A {{age}}-year-old {{sex}} applicant with {{marital_status}} status, working as {{job}}. The request is {{credit_amount}} for {{purpose}} over {{duration}} months, with checking account {{checking_account}} and savings account {{savings_account}}. The person lists credit history {{credit_history}}, employment since {{employment_since}}, {{installment_rate}} installment rate, and {{other_debtors}} other debtors. Current details include residence since {{residence_since}} years, property {{property}}, {{other_installment}} other installment plans, housing {{housing}}, and {{existing_credits}} existing credits. The person supports {{num_people}}, {{telephone}} telephone access, and has {{foreign_worker}} foreign-worker status. The credit application has been evaluated and the decision is {{label}}.
    with: 
        - age: the age of the applicant
        - sex: the gender of the applicant (male or female)
        - marital_status: the marital status of the applicant
        - job: the job/occupation of the applicant
        - credit_amount: the credit amount requested
        - duration: the duration of the credit in months
        - purpose: the purpose of the credit
        - checking_account: the checking account status
        - savings_account: the savings account status
        - credit_history: the credit history of the applicant
        - employment_since: the employment duration
        - installment_rate: the installment rate
        - other_debtors: other debtors or guarantors
        - residence_since: years of residence
        - property: property type
        - other_installment: other installment plans
        - housing: housing type
        - existing_credits: number of existing credits
        - num_people: number of people liable to provide maintenance for
        - telephone: telephone availability
        - foreign_worker: foreign worker status
        - label: the credit risk classification label (0 for good credit risk, 1 for bad credit risk). IMPORTANT: Include this as "good credit risk" or "bad credit risk" at the end of the sentence.
    
Output format:
{{
    "1": "A 24-year-old male applicant... The credit application has been evaluated and the decision is good credit risk.",
    "2": "A 30-year-old female applicant... The credit application has been evaluated and the decision is bad credit risk.",
    "3": "...",
    ...
    "{n_samples}": "..."
}}

Strict requirement: 
- Return exactly {n_samples} entries in the JSON object
- Use numeric keys from "1" to "{n_samples}"
- Each sentence must end with "The credit application has been evaluated and the decision is [good/bad] credit risk."
- Do not include any explanatory text, commentary, or extra fields
- The JSON must be directly parseable
"""
}
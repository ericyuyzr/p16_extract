import torch
import json
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


FEW_SHOT_EXAMPLES = """
#### EXAMPLE 1: CLEAR NEGATIVE (FOLLOW-UP BIOPSY)
Input: "msh ~\& iha-mt-pac... microscopic immunohistochemistry for p16 has been performed and is negative. macroscopic... received is one specimen container..."
Output:
<think>
The report is for an endocervical curettage (ECC) follow-up after a LEEP procedure. 
Under the 'microscopic' section, it explicitly states: 'immunohistochemistry for p16 has been performed and is negative.'
The final diagnosis also confirms 'negative for intraepithelial lesion and malignancy.'
</think>
{
  "status": "negative",
  "rationale": "The microscopic section explicitly states that the p16 immunohistochemistry is negative, supporting the diagnosis of no intraepithelial lesion."
}

### EXAMPLE 2: ADDENDUM WITH POSITIVE RESULT
Input: "addendum diagnosis p16 immunostain was performed and is positive, confirming hsil/cin ii. e"
Output:
<think>
This is a short addendum. It clearly states the p16 immunostain is 'positive'. It correlates this with HSIL/CIN II.
</think>
{
  "status": "positive",
  "rationale": "Directly stated as positive in the addendum to confirm HSIL."
}

### EXAMPLE 3: NEGATIVE (HIGH-GRADE BUT PATCHY)
Input: "...endometrioid adenocarcinoma, figo grade 3 of 3... microscopic description: the tumor cells are strongly positive for vimentin, have patchy staining for p16, and have focal staining for cea."
Output:
<think>
The report is for a Grade 3 endometrioid adenocarcinoma. 
While it is a high-grade tumor, the microscopic description specifically states 'patchy staining for p16'. 
In endometrial pathology, patchy p16 is a 'wild-type' or negative pattern. High grade does not override the patchy staining pattern.
</think>
{"label": "Negative", "reason": "Explicitly described as 'patchy p16', which is a negative/wild-type result regardless of tumor grade."}
"""

model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

def build_prompt(report_text):
    return f"""
You are a pathology extraction AI. Extract the p16 IHC status.
Follow the logical reasoning style of the examples.

{FEW_SHOT_EXAMPLES} # (Include the 4 examples we discussed here)

### NEW TASK ###
Input: "{report_text}"
Output:
"""

def extract(report):
    # Basic cleaning to save tokens
    clean_text = re.sub(r'-{3,}', '', report)
    
    inputs = tokenizer(build_prompt(clean_text), return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=600, temperature=0.6)
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract Reasoning and JSON
    thinking = re.search(r'<think>(.*?)</think>', full_output, re.DOTALL)
    json_body = re.search(r'\{.*\}', full_output, re.DOTALL)
    
    return {
        "thinking": thinking.group(1).strip() if thinking else "N/A",
        "data": json.loads(json_body.group()) if json_body else {"status": "error"}
    }



df = pd.read_csv("balanced_p16_subtable.csv")

labels = []
rationales = []
internal_logics = []

for index, row in df.iterrows():
    print(f"Processing row {index + 1} of {len(df)}...")
    
    try:
        # Pass the specific column 'anonymized_message' to your extract function
        res = extract(row['anonymized_message'])
        
        # Get data safely from the response
        labels.append(res['data'].get('label')) # Use 'label' since we mapped it to 3 groups
        rationales.append(res['data'].get('rationale'))
        internal_logics.append(res['thinking'])
    except Exception as e:
        print(f"Error on row {index}: {e}")
        labels.append("Other")
        rationales.append(f"Error: {str(e)}")
        internal_logics.append("N/A")

df['llm_status'] = labels
df['llm_rationale'] = rationales
df['llm_internal_logic'] = internal_logics

df.to_csv("p16_final_labels_complete.csv", index=False)
print("Saved complete table with all original columns.")

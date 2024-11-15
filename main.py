from fhirpathpy import evaluate
import json

def evaluate_fhirpath(data, fhirpath_expression):
    try:
        result = evaluate(data, fhirpath_expression, [])
        return result
    except Exception as e:
        print(f"Error in evaluating FHIRPath expression: {str(e)}")
        return None

if __name__ == "__main__":
    # Load the JSON file
    file_path = "1000208-ips.json"  # Replace with your file path
    try:
        with open(file_path, 'r') as f:
            patient_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {str(e)}")
        exit(1)

    # List of FHIRPath queries
    queries = [
        ("How old is the patient?", "entry.resource.where(resourceType = 'Patient').birthDate"),
        ("What is the patient's drug allergy?", "entry.resource.where(resourceType = 'AllergyIntolerance' and category = 'medication').code.text"),
        ("What are the patient's current medications?", "entry.resource.where(resourceType = 'MedicationStatement' and status = 'active').medicationCodeableConcept.text"),
        ("What is the patient's gender?", "entry.resource.where(resourceType = 'Patient').gender"),
        ("Does the patient have a history of smoking?", "entry.resource.where(resourceType = 'Observation' and code.coding.display = 'Tobacco smoking status').valueCodeableConcept.text"),
        ("What is the patient's primary diagnosis?", "entry.resource.where(resourceType = 'Condition' and clinicalStatus.coding.code = 'active').code.coding.display"),
        ("What is the patient's contact phone number?", "entry.resource.where(resourceType = 'Patient').telecom.where(system = 'phone').value"),
        ("What are the patient's laboratory test results?", "entry.resource.where(resourceType = 'Observation' and category.coding.code = 'laboratory').valueQuantity.value"),
        ("What is the patient's blood type?", "entry.resource.where(resourceType = 'Observation' and code.coding.display = 'Blood group').valueCodeableConcept.text"),
        ("What is the patient's weight?", "entry.resource.where(resourceType = 'Observation' and code.coding.display = 'Body Weight').valueQuantity.value"),
    ]

    # Iterate through queries and evaluate them
    for question, fhirpath_expression in queries:
        print(f"Question: {question}")
        result = evaluate_fhirpath(patient_data, fhirpath_expression)
        if result:
            print(f"Result: {result}")
        else:
            print("No result found.")
        print("-" * 40)

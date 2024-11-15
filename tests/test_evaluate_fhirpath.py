import pytest
import json
from app_fhir import evaluate_fhirpath

def load_json_file(path):
    with open(path, 'r') as file:
        return json.load(file)

def load_test_data():
    # Assuming you have a test JSON file in resources
    return load_json_file("tests/resources/801941-ips.json")

def test_patient_age():
    data = load_test_data()
    result = evaluate_fhirpath(data, "Bundle.entry.resource.where(resourceType = 'Patient').birthDate")
    print(f"\nPatient birthdate: {result}")
    assert isinstance(result, list), "Should return a list"
    assert len(result) == 1, "Should return a list of 1"
    assert result[0] == '1951-09-30', "Should return a birthdate of 1951-09-30"

def test_patient_food_allergy():
    data = load_test_data()
    result = evaluate_fhirpath(data, "Bundle.entry.resource.where(resourceType = 'AllergyIntolerance').where(category = 'food').code.text")
    print(f"\nFood allergies: {result}")
    assert result == ['Allergy to grass pollen', 'Shellfish allergy'], "Should return allergies of ['Allergy to grass pollen', 'Shellfish allergy']"

def test_patient_current_medications():
    data = load_test_data()
    result = evaluate_fhirpath(data, "Bundle.entry.resource.where(resourceType = 'MedicationRequest').where(status = 'active').medicationCodeableConcept.text")
    print(f"\nCurrent medications: {result}")
    assert  len(result) == 3, "Should return a list of active medications"
    expected_medications = [
        'Alendronic acid 10 MG Oral Tablet',
        '10 ML Furosemide 10 MG/ML Injection',
        'Furosemide 40 MG Oral Tablet'
    ]
    assert result == expected_medications, (
        "Should return medications of "
        f"{expected_medications}"
    )

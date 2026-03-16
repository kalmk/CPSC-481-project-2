from probability4e import *
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import json



T, F = True, False

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class Diagnostics:
    """ Use a Bayesian network to diagnose between three lung diseases """

    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)

        self.network_description = """
        The following is a description of a Bayesian network for diagnosing lung diseases:
        Variables:
            - VisitAsia: whether the patient has visited Asia recently.
            - Smoking: whether the patient is a smoker.
            - TB: whether the patient has tuberculosis.
            - Cancer: whether the patient has lung cancer.
            - Bronchitis: whether the patient has bronchitis.
            - TBorCancer: true if TB or Cancer is true.
            - Xray: whether the chest X-ray is abnormal.
            - Dyspnea: whether the patient has dyspnea.

        Structure:
            - VisitAsia -> TB
            - Smoking -> Cancer
            - Smoking -> Bronchitis
            - TB -> TBorCancer
            - Cancer -> TBorCancer
            - TBorCancer -> Xray
            - TBorCancer -> Dyspnea
            - Bronchitis -> Dyspnea

        Prior probabilities:
            P(VisitAsia=True) = 0.01
            P(VisitAsia=False) = 0.99

            P(Smoking=True) = 0.5
            P(Smoking=False) = 0.5

        Conditional probabilities:
            P(TB=True | VisitAsia=True) = 0.05
            P(TB=True | VisitAsia=False) = 0.01

            P(Cancer=True | Smoking=True) = 0.1
            P(Cancer=True | Smoking=False) = 0.01

            P(Bronchitis=True | Smoking=True) = 0.6
            P(Bronchitis=True | Smoking=False) = 0.3

            P(TBorCancer=True | TB=True, Cancer=True) = 1.0
            P(TBorCancer=True | TB=True, Cancer=False) = 1.0
            P(TBorCancer=True | TB=False, Cancer=True) = 1.0
            P(TBorCancer=True | TB=False, Cancer=False) = 0.0

            P(Xray=True | TBorCancer=True) = 0.99
            P(Xray=True | TBorCancer=False) = 0.05

            P(Dyspnea=True | TBorCancer=True, Bronchitis=True) = 0.9
            P(Dyspnea=True | TBorCancer=True, Bronchitis=False) = 0.7
            P(Dyspnea=True | TBorCancer=False, Bronchitis=True) = 0.8
            P(Dyspnea=True | TBorCancer=False, Bronchitis=False) = 0.1

        Important:
            - TBorCancer is a deterministic OR node.
            - The final diagnosis must choose the single most likely disease among:
            TB, Cancer, Bronchitis.
        """

    def diagnose (self, asia, smoking, xray, dyspnea):
        # 1. Build the prompt
        # 2. Call the Gemini API
        # 3. Parse the JSON response
        # 4. Return the [disease, probability]
        prompt = f"""You are solving a Bayesion Network disease diagnosis problem.
            The possible diseases are: TB, Cancer, Bronchitis.
            Use the Bayesian network below and the evidence to compute the most likely disease and its probability.
            Return your answer in the format: [disease, probability], where disease is one of "TB", "Cancer", "Bronchitis" and probability is a number between 0 and 1. Use JSON format.
            Bayesian Network: {self.network_description}

            Evidence:
            - VisitAsia: {asia}
            - Smoking: {smoking}
            - Xray: {xray}
            - Dyspnea: {dyspnea}
            """
        
        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema={
                    "type": "object",
                    "properties": {
                        "disease": {
                            "type": "string",
                            "enum": ["TB", "Cancer", "Bronchitis"]
                        },
                        "probability": {
                            "type": "number"
                        }
                    },
                    "required": ["disease", "probability"]
                }
            )
        )

        data = json.loads(response.text)
        disease = data["disease"]
        probability = data["probability"]
        return [disease, probability]

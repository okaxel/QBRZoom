"""
QBRZoom - The Quarterly Business Review Risk Assessment Tool
============================================================

Submodule: engine
-----------------

This file contains the analytical engine.
"""

from dataclasses import dataclass
from json import dumps as json_dumps, JSONDecodeError, loads as json_loads

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from .data import FineProcessedData, Team


LABEL_RISK_FACTORS = 'risk_factors'
LABEL_RISK_SCORE = 'risk_score'
MODEL_ID = 'meta-llama/Llama-2-7b-chat-hf'


@dataclass(kw_only=True, frozen=True)
class RawLLMResponse:
    """Raw LLM response container.
    """


    prompt: str
    response: dict | None


@dataclass(kw_only=True, frozen=True)
class RiskItem:
    """Message level risk prediction container.
    """


    message_id: int
    message: str
    score: float
    factors: list[str]


@dataclass(kw_only=True, frozen=True)
class ProjectRiskReport:
    """Project level risk prediction container.
    """


    project_id: int
    messages: list[RiskItem] | None
    top_score: float
    average_score: float
    non_zero_average: float
    non_zero_count: int
    factors: list[str]
    error_message: str | None


@dataclass(kw_only=True, frozen=True)
class FinalRiskReport:
    """Risk report data.
    """


    team: Team
    projects: list[ProjectRiskReport]


class HuggingFaceTool:
    """Hugging Face Management Toolkit.
    """


    __gpu_enabled: bool = True
    __pipeline: any = None
    __role: str = 'precise classifier'
    __tokenizer: any = None


    @classmethod
    def disable_gpu(cls) -> None:
        """Disable the use of GPU for the pipeline.
        """

        cls.__gpu_enabled = False


    @classmethod
    def enable_gpu(cls) -> None:
        """Enable the use of GPU for the pipeline.
        """

        cls.__gpu_enabled = True


    @classmethod
    def get_email_risk(cls, email_texts: list[str]) -> RawLLMResponse:
        """Get message flow risk prediction from local LLM model.

        Parameters
        ----------
        email_texts : list[str] (Positional-only)
            Text contents of the messages.

        Returns
        -------
        RiskReport | None
            The created prompt and the raw risk prediction response.
        """
        
        cls.init_pipeline()
        prompt_ = (f"You are {add_indefinite_article(cls.__role)}. "
                   "Given a list of inbound emails, return a JSON object where "
                   "each key is the index of the email in the input list, and "
                   "each value is an object with:\n"
                   "  - risk_score: a number between 0.0 (no risk) and 1.0 (high risk)\n"
                   "  - risk_factors: an array of short strings explaining why the email is risky.\n\n"
                   f"Input emails:\n{json_dumps(email_texts, indent=2)}\n\n"
                   "Output JSON:")
        response_ = cls.__pipeline(prompt_, do_sample=False)[0]['generated_text']
        try:
            raw_data_ = json_loads(response_)
        except JSONDecodeError:
            print('[ERROR] Failed to decode response as clear JSON data.')
            raw_data_ = fallback_json_decode(response_)
            if raw_data_ is None:
                print('[ERROR] Failed to decode response as noisy JSON data.')
                return RawLLMResponse(prompt=prompt_, response=None)
        return RawLLMResponse(prompt=prompt_, response=raw_data_)


    @classmethod
    def get_role(cls) -> str:
        """Get role for email risk prediction. (Read-only)

        Returns
        -------
        str
            Get the role that is sent to the Large Language Model for risk
            prediction.
        """

        return cls.__role


    @classmethod
    def init_pipeline(cls) -> None:
        """Initialize pipeline if needed.
        """

        if cls.__pipeline is None:
            cls.__tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model_ = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                                          load_in_4bit=True,
                                                          device_map='auto',
                                                          quantization_config={
                                                              'bnb_4bit_compute_dtype': 'float16',
                                                              'bnb_4bit_use_double_quant': True,
                                                              'bnb_4bit_quant_type': 'nf4'
                                                              }
                                                          )
            cls.__pipeline = pipeline('text2text-generation',
                                      model=model_,
                                      device=0 if cls.__gpu_enabled else 1)


    @classmethod
    def is_gpu_enabled(cls) -> bool:
        """Get whether GPU use is enabled or not.

        Returns
        -------
        bool
            True if enabled, False if not.
        """

        return cls.__gpu_enabled


    @classmethod
    def set_role(cls, new_role: str, /) -> None:
        """Set the role for the email risk prediction request.

        Parameters
        ----------
        new_role : str (Positional-only)
            The desired role that is sent to the Large Language Model for risk
            prediction.
        """

        cls.__role = new_role


def add_indefinite_article(word: str, /) -> str:
    """Append string with an indefinite article.

    Parameters
    ----------
    word : str (Positional-only)
        The string to append.

    Returns
    -------
    str
        The string with an indefinite article "a" or "an" depending on the given
        string.
    """

    return f'a {word}' if not begins_with_vowel(word) else f'an {word}'


def assembly_project_risk_report(project_id: int, response: dict,
                                 messages: list[str], /) -> ProjectRiskReport:
    """Create project level risk report.

    Parameters
    ----------
    project_id : int (Positional-only)
        The id of the project.
    response : dict (Positional-only)
        Raw response data.
    messages : list[str] (Positional-only)
        Texts of the original messages.

    Returns
    -------
    ProjectRiskReport
        The project level risk report.
    """

    error_ = ''
    invalid_data_type_at_ = []
    invalid_inner_data_types_at_ = []
    missing_keys_at_ = []
    missing_labels_at_ = []
    scores_ = []
    factors_ = []
    items_ = []
    for i in range(len(messages)):
        key_ = str(i)
        if key_ not in response:
            missing_keys_at_.append(key_)
            continue
        if not isinstance(response[key_], dict):
            invalid_data_type_at_.append(key_)
            continue
        if LABEL_RISK_SCORE not in response[key_]\
                                    or LABEL_RISK_FACTORS not in response[key_]:
            missing_labels_at_.append(key_)
            continue
        if not isinstance(response[key_][LABEL_RISK_SCORE], float)\
                    or not isinstance(response[key_][LABEL_RISK_FACTORS], list):
            invalid_inner_data_types_at_.append(key_)
            continue
        scores_.append(response[key_][LABEL_RISK_SCORE])
        factors_ += response[key_][LABEL_RISK_FACTORS]
        items_.append(RiskItem(message_id=i + 1, message=messages[i],
                               score=response[key_][LABEL_RISK_SCORE],
                               factors=response[key_][LABEL_RISK_FACTORS][:]))
    length_ = len(scores_)
    sum_ = sum(scores_) if length_ > 0 else 0.0
    max_ = max(scores_) if length_ > 0 else 0.0
    non_zero_count_ = sum([1 for s in scores_ if s > 0])
    return ProjectRiskReport(project_id=project_id, messages=items_,
                             top_score=max_,
                             average_score=sum_ / length_,
                             non_zero_average=sum_ / non_zero_count_ if non_zero_count_ > 0 else 0.0,
                             non_zero_count=non_zero_count_,
                             factors=list(sorted(set(factors_))) if len(factors_) > 0 else [],
                             error_message=None if error_ == '' else error_)


def begins_with_vowel(word: str, /) -> bool:
    """Detect if the word begins with vowel or not.

    Parameters
    ----------
    word : str (Positional-only)
        The word to check.

    Returns
    -------
    bool
        True if the word begins with vowel, False or not.
    """

    return word[0].lower() in ['a', 'e', 'i', 'o', 'u']


def fallback_json_decode(content: str, /) -> dict | list | None:
    """Decode JSON from a noisy string.

    Parameters
    ----------
    content : str (Positional-only)
        The content to search JSON in.

    Returns
    -------
    dict | list | None
        The retrieved JSON object or None if no JSON is detected.
    """

    for bracket in ['{}', '[]']:
        begin_ = content.find(bracket[0])
        end_ = content.rfind(bracket[1])
        if 0 <= begin_ < end_:
            try:
                return content[begin_:end_ + 1]
            except JSONDecodeError:
                continue
    return None


def llm_risk_detector(data: FineProcessedData, /) -> FinalRiskReport:
    """Perform LLM based risk detections for all the projects.

    Parameters
    ----------
    data : FineProcessedData (Positional-only)
        Tha fine processed data.

    Returns
    -------
    FinalRiskReport
        The final risk report.
    """

    risk_reports_: list[ProjectRiskReport] = []
    for project_id, messages in enumerate(data.data, 1):
        texts_ = [m.protected_text for m in messages]
        raw_results_ = HuggingFaceTool.get_email_risk(texts_)
        if not isinstance(raw_results_.response, dict):
            error_ = f'API request led to a {type(raw_results_.response)} response. Expected is dict.'
            print(f'[ERROR] {error_}')
            risk_reports_.append(ProjectRiskReport(project_id=project_id,
                                                   messages=None, top_score=0.0,
                                                   average_score=0.0,
                                                   non_zero_average=0.0,
                                                   non_zero_count=0, factors=[],
                                                   error_message=error_))
            continue
        risk_reports_.append(assembly_project_risk_report(project_id,
                                                          raw_results_.response,
                                                          messages))
    return FinalRiskReport(team=data.team, projects=risk_reports_)
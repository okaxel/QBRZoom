"""
QBRZoom - The Quarterly Business Review Risk Assessment Tool
============================================================

Submodule: data
---------------

This file contains the data pipeline of data ingestion, processing, feature
extraction and enrichment.
"""

from __future__ import annotations # Imported only to support static method's typing
from dataclasses import dataclass
from datetime import timezone
from os.path import isfile, join as path_join
from re import compile as re_compile, VERBOSE as re_VERBOSE
from unicodedata import category as unicode_category, normalize as unicode_normalize

from dateparser import parse as date_parse
from dateparser.search import search_dates
from sentence_transformers import SentenceTransformer
from spacy import load as spacy_load
from spacy.language import Language
from spacy.tokens.doc import Doc
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


COLLEAGUE_PATTERN = re_compile(r'''
                               ^                       # start of string
                               (?P<role>[^:]+)         # "role" = everything up to the first colon
                               :\s*                    # colon + optional space
                               (?P<name>[^()]+)        # "name" = text up to the opening parenthesis
                               \s*\(                   # optional space + literal "("
                               (?P<email>[^)]+)        # "email" = text inside parentheses
                               \)\s*$                  # closing ")" + optional space + end of string
                               ''', re_VERBOSE)
DATA_ROOT = path_join('.', 'mail')
EMAIL_FOOTER_PATERN = re_compile(r'\n--\s*\n')
EMAIL_HISTORY_PATTERN = re_compile(r'\n--\s*\n')
EMAIL_PATTERN = re_compile(r'[a-zA-Z0-9_.+\-À-ÖØ-öø-ÿ]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-\.]+')
EMAIL_PRIVACY_PATTERN = re_compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
EMAIL_PRIVACY_PLACEHOLDER = '<EMAIL_REDACTED>'
EMAIL_PREFIX = 'email'
HEADER_PATTERN = re_compile(r'^([!-9;-~]+):\s*(.*)$')
PATH_TO_COLLEAGUES = path_join(DATA_ROOT, 'Colleagues.txt')
PERSONAL_PRIVACY_PLACEHOLDER = '<NAME_REDACTED>'


@dataclass(kw_only=True, frozen=True)
class Colleague:
    """Colleague data
    """


    role: str
    name: str
    email: str


    @staticmethod
    def from_line(line: str, /) -> Colleague | None:
        """Crate Colleague object from text.
        
        Parameters
        ----------
        line : str (Positional-only)
            Line from the file.
            
        Returns
        -------
        Colleague | None
            The Colleague object in case of success, None in case of failure.
        """

        data_ = COLLEAGUE_PATTERN.match(line.strip())
        if data_ is None:
            return None
        parameters_ = {k : v for k, v in data_.groupdict().items()}
        parameters_['email'] = plain_latin(parameters_['email'].replace('"', ''))
        return Colleague(**parameters_)


class LanguageTools:
    """Language tooling singleton
    """


    __sentiment_analyzer: SentimentIntensityAnalyzer | None = None
    __embedding_tool: SentenceTransformer | None = None
    __nlp_tool: Language | None = None


    @classmethod
    def embedding_tool(cls) -> SentenceTransformer:
        """Get the singleton sentence transformer (embedding creator) instance.

        Returns
        -------
        SentenceTransformer
            The sentence transformer instance.
        """

        if cls.__embedding_tool is None:
            cls.__embedding_tool = SentenceTransformer('all-MiniLM-L6-v2')
        return cls.__embedding_tool


    @classmethod
    def nlp(cls, text: str, /) -> Doc:
        """Perform NLP processing with the singleton NLP tool.

        Parameters
        ----------
        text : str (Positional-only)
            Text to analyze.

        Returns
        -------
        Doc
            The results of the analysis.
        """

        if cls.__nlp_tool is None:
            cls.__nlp_tool = spacy_load('en_core_web_sm')
        return cls.__nlp_tool(text)


    @classmethod
    def sentiment_analyzer(cls) -> SentimentIntensityAnalyzer:
        """Get the singleton analyzer instance.

        Returns
        -------
        SentimentIntensityAnalyzer
            The analyzer instance.
        """

        if cls.__sentiment_analyzer is None:
            cls.__sentiment_analyzer = SentimentIntensityAnalyzer()
        return cls.__sentiment_analyzer


class RawMessage:
    """Email content
    """


    __REQUIRED_HEADER_KEYS = ['from', 'to', 'subject', 'date']


    def __init__(self) -> None:
        """Create a RawMessage instance.
        """

        self.__headers: dict[str, str] = {}
        self.__body: str = ''


    def add_header(self, line: str, /, avoid_overwrite: bool = True) -> bool:
        """Add header.

        Parameters
        ----------
        line : str (Positional-only)
            Raw text line that contains header data.
        avoid_overwrite : bool, optional (Keyword-only)
            Whether or not to avoid the overwriting already existing header. It
            defaults to True.

        Returns
        -------
        bool
            _description_
        """

        data_ = HEADER_PATTERN.match(line)
        if data_ is None:
            return False
        key_ = data_.group(1).strip().lower()
        if avoid_overwrite and key_ in self.__headers:
            return False
        self.__headers[key_] = data_.group(2).strip()
        return True


    @property
    def body(self) -> str:
        """Get message body.

        Returns
        -------
        str
            The body of the message.
        """

        return self.__body


    def has_body(self) -> bool:
        """Check if message has body or not.

        Returns
        -------
        bool
            True if message body is not empty, else False.
        """

        return len(self.__body.strip()) > 0


    def has_required_headers(self) -> bool:
        """Check if each required field of message is set.

        Returns
        -------
        bool
            True if all fields are set, else False.
        """

        return all([k in self.__headers for k in self.__REQUIRED_HEADER_KEYS])


    def headers(self) -> dict[str, str]:
        """Get a copy of the headers dictionary.

        Returns
        -------
        dict[str, str]
            The copied headers.

        Notes
        -----
        Any modification of the returned object do not affect the message
        instance.
        """

        return {k.capitalize() : v for k, v in self.__headers.items()}


    @staticmethod
    def is_header_alike(line: str, /) -> bool:
        """Check whether the given text can be qualified as a header line or not.

        Parameters
        ----------
        line : str (Positional-only)
            The text to check.

        Returns
        -------
        bool
            True if the text is considerable as a potential header data source,
            else False.
        """

        return HEADER_PATTERN.match(line) is not None


    def sender_email(self) -> str:
        """Get the email address of the sender.

        Returns
        -------
        str
            The email address of the sender or empty string. An empty string is
            given if the message does not have a "From" header or the content of
            the "From" header does not contain any text that matches a basic
            email address pattern.
        """

        if 'from' not in self.__headers:
            return ''
        data_ = EMAIL_PATTERN.search(plain_latin(self.__headers['from'].replace('"', '')))
        if data_ is None:
            return ''
        return data_[0]


    def set_body(self, lines: list[str], /) -> None:
        """Set message body.

        Parameters
        ----------
        lines : list[str] (Positional-only)
            Content of the message body.
        """

        self.__body = ' '.join(lines)


    def timestamp(self) -> int:
        """Get message date as a second based timestamp.

        Returns
        -------
        int
            The timestamp based on the "Date" header or -1 if the "Date" header
            does not exist or its content is not available to convert to a
            datetime object.
        """

        if 'date' not in self.__headers:
            return -1
        try:
            date_ = date_parse(self.__headers['date'],
                            settings={'RETURN_AS_TIMEZONE_AWARE' : True,
                                        'TO_TIMEZONE' : 'UTC'})
        except Exception:
            return -1
        if date_.tzinfo is None:
            date_ = date_.replace(tzinfo=timezone.utc)
        return int(date_.astimezone(timezone.utc).timestamp())
        


class Team:
    """Team of colleagues
    """


    def __init__(self, members: list[Colleague] | None = None, /):
        """Create a Team instance.

        Parameters
        ----------
        members : list[Colleague] | None, optional (Positional-only)
            List of initial team members or None to create an empty team. It
            defaults to None.
        """

        self.__members: list[Colleague] = []
        if members is not None:
            self.__members = list(members) # Same object, new wrapper.
        self.__names: set[str] = set()


    def add_member(self, new_member: Colleague, /):
        """Add member if not already added.

        Parameters
        ----------
        new_member : Colleague (Positional-only)
            The member to add.
        Notes
        -----
        Already existing membership is checked by the email address.
        """

        for m in self.__members:
            if new_member.email == m.email:
                return
        self.__members.append(new_member)
        for name in new_member.name:
            self.__names.add(name)


    @property
    def count(self) -> int:
        """Count of team members. (Read-only)

        Returns
        -------
        int
            The count of registered team members.
        """

        return len(self.__members)


    def get(self, index: int, /) -> Colleague | None:
        """Get team member by id.

        Parameters
        ----------
        index : int (Positional-only)
            The id of the colleague

        Returns
        -------
        Colleague | None
            Colleague object in case if id exists, else None.
        """

        if -1 < index < len(self.__members):
            return self.__members[index]


    def get_id_by_email(self, email: str, /) -> int:
        """Get team member id by email.

        Parameters
        ----------
        email : str (Positional-only)
            Email address of the 

        Returns
        -------
        int
            A non-negative id number in case of success, else -1.
        """

        for i, member in enumerate(self.__members):
            if member.email == email:
                return i
        return -1


    @property
    def members(self) -> list[Colleague]:
        """Get a copied list of colleagues.

        Returns
        -------
        list[Colleague]
            The copied list of all colleagues

        Notes
        -----
        Changes made om the returned list do not affect the Team object itself.
        """

        return list(self.__members)


    def redact_names(self, text: str, /) -> str:
        """Replace team member names with privacy placeholder.

        Parameters
        ----------
        text : str (Positional-only)
            The text to protect.

        Returns
        -------
        str
            The protected text.
        """

        result_ = text
        for name in self.__names:
            result_.replace(name, PERSONAL_PRIVACY_PLACEHOLDER)
        return result_



    def __repr__(self) -> str:
        """Unambiguous string representation of the instance.

        Returns
        -------
        str
            Valid Python expression that can be used to recreate this instance
            via eval().
        """

        members_ = ', '.join(repr(m) for m in self.__members)
        return f'Team([{members_}])'


    def __str__(self) -> str:
        """Human readable information about the instance.

        Returns
        -------
        str
            Some information about the instance.
        """

        return f'Team of {len(self.__members)} colleague(s).'


@dataclass(kw_only=True, frozen=True)
class PreProcessedData:
    """Data class for ingested and pre-processed data.
    """

    threads: list[list[RawMessage]]
    team: Team


class MessageWithFeatures:
    """Message object with extracted and enriched features and data
    """


    def __init__(self, raw_message: RawMessage, team: Team, /, *,
                 key_phrases_target: int = 3) -> None:
        """Initialize a MessageWithFeatures instance.

        Parameters
        ----------
        raw_message : RawMessage (Positional-only)
            RawMessage object to get raw data from.
        team : Team (Positional-only)
            Team object to identify message senders.
        key_phrases_target : int, optional (Keyword-only)
            Number of selected key phrases, by default 10.
        """

        self.__raw_text = raw_message.body
        self.__timestamp = raw_message.timestamp()
        self.__colleague_id = team.get_id_by_email(raw_message.sender_email())
        doc_ = LanguageTools.nlp(self.__raw_text)
        self.__key_phrases_target = key_phrases_target
        self.__sentences: list[str] = [s.text for s in doc_.sents]
        text_ = ' '.join(self.__sentences)
        self.__tokens: list[str] = [t.text for t in doc_]
        self.__entities: list[tuple[str, str]] = [(e.text, e.label) for e in doc_.ents]
        self.__key_phrases: list[str] = [c.text for c in doc_.noun_chunks][:key_phrases_target]
        self.__sentiment: dict[str, float] = LanguageTools.sentiment_analyzer().polarity_scores(text_)
        self.__embedding: list = LanguageTools.embedding_tool().encode(text_).tolist()
        self.__mentioned_dates: list[str] = extract_dates(text_)
        self.__protected_text = simple_privacy_protector(self.__raw_text)
        for entity in reversed(doc_.ents):
            if entity.label_ == 'PERSON':
                start_, end_ = entity.start_char, entity.end_char
                self.__protected_text = self.__protected_text[:start_]\
                                        + PERSONAL_PRIVACY_PLACEHOLDER\
                                        + self.__protected_text[end_:]
        self.__protected_text = team.redact_names(self.__protected_text)


    @property
    def colleague_id(self) -> int:
        """Get the id of the colleague.

        Returns
        -------
        int
            The id of the colleague.
        """

        return self.__colleague_id


    @property
    def embedding(self) -> list:
        """Get created embeddings.

        Returns
        -------
        list
            The list of embeddings.
        """

        return list(self.__embedding)


    @property
    def entities(self) -> list[tuple[str, str]]:
        """Get list of entities.

        Returns
        -------
        list[tuple[str, str]]
            The list of entities.
        """

        return list(self.__entities)


    @property
    def key_phrases(self) -> list[str]:
        """Get key phrases.

        Returns
        -------
        list[str]
            List of strings containing the key phrases.
        """

        return list(self.__key_phrases)


    @property
    def key_phrases_target(self) -> int:
        """Get the desired key phrases count.

        Returns
        -------
        int
           The targeted key phrases count.
        """

        return self.__key_phrases_target


    @property
    def mentioned_dates(self) -> list[str]:
        """Get date mentions from text.

        Returns
        -------
        list[str]
            List of ISO formatted data strings.
        """

        return list(self.__mentioned_dates)


    @property
    def protected_text(self) -> str:
        """Get the privacy protected text.

        Returns
        -------
        str
            The privacy protected string.
        """

        return self.__protected_text


    @property
    def raw_text(self) -> str:
        """Get the original raw text.

        Returns
        -------
        str
            The original initializer string.
        """

        return self.__raw_text


    @property
    def sender_is_team_member(self) -> bool:
        """Get whether the sender is a team member or not.

        Returns
        -------
        bool
            The id of the colleague or -1 if the sender is not a team member.
        """

        return self.__colleague_id != -1


    @property
    def sentence_count(self) -> int:
        """Get count of sentences.

        Returns
        -------
        int
            Number of sentences stored in the message.
        """

        return len(self.__sentences)


    @property
    def sentences(self) -> list[str]:
        """Get sentences.

        Returns
        -------
        list[str]
            List of strings containing sentences.
        """

        return list(self.__sentences)


    @property
    def sentiment(self) -> dict[str, float]:
        """Get sentiment analysis' results.

        Returns
        -------
        dict[str, float]
            List of sentiment values.
        """

        return dict(self.__sentiment)

    @property
    def timestamp(self) -> int:
        """Get date of the message.

        Returns
        -------
        int
            Seconds based timestamp.
        """

        return self.__timestamp


    @property
    def token_count(self) -> int:
        """Get count of tokens.

        Returns
        -------
        int
            The count of tokens in the list.
        """

        return len(self.__tokens)


    @property
    def tokens(self) -> list[str]:
        """Get tokens.

        Returns
        -------
        list[str]
            List of tokens.
        """

        return list(self.__tokens)


@dataclass(kw_only=True, frozen=True)
class FineProcessedData:
    """Data class for ingested and pre-processed data.
    """

    team: Team
    data: list[list[MessageWithFeatures]]


def data_ingestion_and_preprocessing() -> PreProcessedData:
    """Perform data ingestion and pre-processing.

    Returns
    -------
    PreProcessedData
        The results of the stage.
    """

    team_ = get_team()
    raw_thread_paths_ = get_email_paths()
    raw_thread_data_ = []
    for path in raw_thread_paths_:
        raw_thread_data_.append(get_raw_thread_data(path))
    return PreProcessedData(threads=raw_thread_data_, team=team_)


def data_pipeline() -> FineProcessedData:
    """Perform data pipeline actions.

    Returns
    -------
    FineProcessedData
        All the processed data.
    """

    pre_processed_ = data_ingestion_and_preprocessing()
    return FineProcessedData(team=pre_processed_.team,
                             data=feature_extraction_and_enrichment(pre_processed_))


def extract_dates(text: str, /) -> list[str]:
    """Extract future date alike mentions.

    Parameters
    ----------
    text : str (Positional-only)
        The message text to search future date mentions.

    Returns
    -------
    list[str]
        List of ISO formatted date strings.
    """

    result_: list[str] = []
    data_ = search_dates(text, settings={'PREFER_DATES_FROM': 'future'})
    if data_ is None:
        return result_
    for entry in data_:
        result_.append(entry[1].isoformat())
    return result_


def feature_extraction_and_enrichment(data: PreProcessedData, /) -> list[list[MessageWithFeatures]]:
    """Extract features and enrich data of messages.

    Parameters
    ----------
    data : PreProcessedData (Positional-only)
        Data about the team and of the processed messages.

    Returns
    -------
    list[list[MessageWithFeatures]]
        Messages with features grouped by threads.
    """

    print(f'[INFO] Team has {data.team.count} member(s).')
    thread_count_ = len(data.threads)
    total_message_count_ = sum(len(t) for t in data.threads)
    print(f'[INFO] There are {thread_count_} thread(s) with {total_message_count_} message(s).')
    result_ = []
    for t, thread in enumerate(data.threads, 1):
        thread_data_ = []
        thread_length_ = len(thread)
        for m, message in enumerate(thread, 1):
            print(f'\rProcessing {m:2d}/{thread_length_:2d} message of {t:2d}/{thread_count_:2d} thread.', end='')
            thread_data_.append(MessageWithFeatures(message, data.team))
        result_.append([m for m in thread_data_])
    print(f'\r[INFO] Finished the processing of {thread_count_} thread(s) with {total_message_count_} message(s).')
    return result_
    


def get_email_paths(*, include_path: bool = True) -> list[str]:
    """Get the list of the email paths.

    Parameters
    ----------
    include_path : bool, optional (Keyword-only)
        Whether to include full relative path or not. Defaults to True.

    Returns
    -------
    list[str]
        Sorted list of email file name(s) or path(s).
    """

    index_ = 1
    current_file_ = f'email{index_}.txt'
    current_path_ = path_join(DATA_ROOT, current_file_)
    paths_ = []
    while isfile(current_path_):
        paths_.append(current_path_ if include_path else current_file_)
        index_ += 1
        current_file_ = f'email{index_}.txt'
        current_path_ = path_join(DATA_ROOT, current_file_)
    return paths_


def get_raw_thread_data(file_path: str, /) -> list[RawMessage]:
    """Read text data and transform it to a list of raw messages.

    Parameters
    ----------
    file_path : str (Positional-only)
        Path of the text content.

    Returns
    -------
    list[RawMessage]
        List of RawMessage objects.
    """

    with open(file_path, 'r', encoding='utf_8') as in_stream:
        lines_ = [l.strip() for l in in_stream.readlines()]
    return split_emails(lines_)


def get_team() -> Team:
    """Get list of all colleagues as a team.

    Returns
    -------
    Team
        The created team.
    """

    team_ = Team()
    with open(PATH_TO_COLLEAGUES, 'r', encoding='utf_8') as in_stream:
        for line in in_stream.readlines():
            colleague_ = Colleague.from_line(line)
            if colleague_ is None:
                continue
            team_.add_member(colleague_)
    return team_


def plain_latin(text: str, /) -> str:
    """Convert text to plain ASCII base characters.

    Parameters
    ----------
    text : str (Positional-only)
        The text to convert.

    Returns
    -------
    str
        The converted text.
    """

    return ''.join(c for c in unicode_normalize('NFD', text)
                   if not unicode_category(c).startswith('M'))


def simple_privacy_protector(text: str, /) -> str:
    """Protect employees privacy.

    Parameters
    ----------
    text : str (Positional-only)
        The text to apply privacy protection.

    Returns
    -------
    str
        The privacy protected test.
    """

    return EMAIL_PRIVACY_PATTERN.sub(EMAIL_PRIVACY_PLACEHOLDER, text)


def split_emails(content: list[str], /) -> list[RawMessage]:
    """Transform email thread to raw messages.

    Parameters
    ----------
    content : list[str] (Positional-only)
        List of stripped text lines.

    Returns
    -------
    list[RawMessage]
        List of RawMessage objects.

    Notes
    -----
    This processing is far from level. It handles the characteristics of the
    given test data. Replace it with a solution that implements the commonly
    used email output forms to get a much more commonly useable and robust
    solution.
    """

    messages_: dict[int, RawMessage] = {}
    current_message_ = RawMessage()
    body_ = []
    for i, line in enumerate(content, 1):
        if len(line) == 0:
            continue
        if not RawMessage.is_header_alike(line):
            body_.append(line)
            continue
        if current_message_.add_header(line):
            continue
        if len(body_) == 0:
            print(f'[ERROR] Dropping empty message right before line {i}.')
        current_message_.set_body(body_)
        timestamp_ = current_message_.timestamp()
        if timestamp_ == -1:
            print(current_message_.headers().keys())
            print(f'[ERROR] Dropping message without date right before line {i}.')
        messages_[timestamp_] = current_message_
        current_message_ = RawMessage()
        body_ = []
        current_message_.add_header(line)
    current_message_.set_body(body_)
    timestamp_ = current_message_.timestamp()
    if current_message_.has_body() and current_message_.has_required_headers()\
                                   and timestamp_ > -1:
        messages_[timestamp_] = current_message_
    result_: list[RawMessage] = []
    for key in sorted(messages_.keys()):
        result_.append(messages_[key])
    return result_
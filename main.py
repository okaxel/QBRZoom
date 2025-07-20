"""
QBRZoom - The Quarterly Business Review Risk Assessment Tool
============================================================

Entry point
-----------
"""

#"""
# Remove the in-line comment symbol "#" from the previous line to use global
# cache instead of project level cache for the modules.

from qbrz.project import set_project_cache
set_project_cache()

#"""

from qbrz.data import data_pipeline
from qbrz.engine import llm_risk_detector
from qbrz.user_interface import create_report


def main() -> None:
    """Entry point.
    """

    processed_data_ = data_pipeline()
    report_ = llm_risk_detector(processed_data_)
    report_text_ = create_report(report_)
    with open('final_report.md', 'w', encoding='utf_8') as out_stream:
        out_stream.write(f'{report_text_}\n')
    print(report_text_)


if __name__ == '__main__':
    main()
else:
    raise Warning('This file is rather a script than a module.')
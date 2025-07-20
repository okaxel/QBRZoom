"""
QBRZoom - The Quarterly Business Review Risk Assessment Tool
============================================================

Submodule: engine
-----------------

This file contains the analytical engine.
"""

from .engine import FinalRiskReport


def create_report(report: FinalRiskReport, /) -> str:
    """Create textual report.

    Parameters
    ----------
    report : FinalRiskReport (Positional-only)
        The data to create report from.

    Returns
    -------
    str
        The textual report in basic markdown format.
    """

    projects_ = ''
    for project in report.projects:
        projects_ += (
            f'## Project #{project.project_id:02d}\n\n'
            f'This project contains {len(project.messages)} message(s).\n\n'
            )
        if project.non_zero_count > 0:
            projects_ += (
                f'Messages with risks: {project.non_zero_count}\n\n'
                f'Average risk factor: **{project.non_zero_average:0.2f}** (non weighted: {project.average_score:0.2f})\n\n'
                )
            if len(project.factors) > 0:
                projects_ += '### Risk factors:\n\n'
                for factor in project.factors:
                    projects_ += f'- {factor}\n'
                projects_ += '\n'
        else:
            projects_ += '**No risk detected.**\n\n'
        if project.error_message is not None:
            errors_ = project.error_message.split('\n')
            projects_ += '### Error messages:\n\n'
            for error in errors_:
                projects_ += f'- {error}\n'
            projects_ += '\n'
    return (
        '# FINAL RISK REPORT\n\n'
        f'Processed {len(report.projects)} projects performed by a team of {report.team.count}.\n\n'
        f'{projects_}\n\n'
        )

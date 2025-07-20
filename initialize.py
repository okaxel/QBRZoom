"""
QBRZoom - The Quarterly Business Review Risk Assessment Tool
============================================================

Project initializer
-------------------

Run this code once after downloading the package.
"""

#"""
# Remove the in-line comment symbol "#" from the previous line to use global
# cache instead of project level cache for the modules.

from qbrz.project import set_project_cache
set_project_cache()

#"""

from huggingface_hub import login

from qbrz.engine import HuggingFaceTool


def main() -> None:
    """Entry point of the initializer.
    """

    token_ = input('Please enter your Hugging Face login token and hit enter.')
    login(token=token_)
    print('[INFO] Please wait while the model gets downloaded.')
    HuggingFaceTool.init_pipeline()
    print('[INFO] Model downloaded. The software is initialized.')


if __name__ == '__main__':
    main()
else:
    raise Warning('This file is rather a script than a module.')
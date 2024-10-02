# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#        Copyright (c) -2022 - Mtumbuka F.                                                    #
#        All rights reserved.                                                                       #
#                                                                                                   #
#        Redistribution and use in source and binary forms, with or without modification, are       #
#        permitted provided that the following conditions are met:                                  #    
#        1. Redistributions of source code must retain the above copyright notice, this list of     #
#           conditions and the following disclaimer.                                                #
#        2. Redistributions in binary form must reproduce the above copyright notice, this list of  #
#           conditions and the following disclaimer in the documentation and/or other materials     #
#           provided with the distribution.                                                         #
#                                                                                                   #
#        THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY      #
#        EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF    #
#        MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE #
#        COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,   #
#        EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF         #
#        SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)     #
#        HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR   #
#        TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS         #
#        SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                               #
#                                                                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


__license__ = "BSD-2-Clause"
__version__ = "2022.1"
__date__ = "28 Jul 2022"
__author__ = "Frank M. Mtumbuka"
__maintainer__ = "Frank M. Mtumbuka"
__email__ = "fmtumbuka@gmail.com"
__status__ = "Development"


from distutils.core import setup


# Read the long description from Readme File

long_description = open("README.md").read()

setup(
    author="Frank Martin Mtumbuka",
    author_email="fmtumbuka@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3.8"
    ],
    copyright="Copyright (c) 2022 - Mtumbuka F. M",
    data_files=[
        (".", ["LICENSE", "README.md"])
    ],
    description="Provide project specific description here.",
    install_requires=[
        "argmagic>=2017.1",
        "expbase",
        "gensim>=3.7.2",
        "insanity>=2017.1",
        "numpy>=1.16.0",
        "scikit-learn>=0.21.1",
        "spacy>=2.1.3",
        "staticinit>=2017.1",
        "tensorboardX>=1.6",
        "tensorflow>=1.13.1",
        "torch>=1.1.0",
        "transformers>=2.1.1"
    ],
    license="BSD-2-Clause",
    long_description=long_description,
    name="noie",
    package_dir={"": "src/main/python"},
    packages=[
            # List all packages here
    ],
    python_requires=">=3.7",
    url="url to the repo",
    version="2022.1"
    )

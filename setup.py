import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()
	
setuptools.setup(
	name="aawedha",
	version="0.1.0",
	author="Okba Bekhelifi",
	author_email="okba.bekhelifi@univ-usto.dz",
	description="",
	long_description=long_description,
	long_description_content_type="text/markdown",	
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3.7",
		"Licence :: OSI Approved :: GNU General Public License v3.0",
		"Operating System :: OS Independent",
	]
)
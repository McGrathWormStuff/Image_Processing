1. Feature Extraction: LiveBodyAnalysis.py and SegmentAnalysis.py: LiveBodyAnalysis.py User Inputs
	a. Absolute path of the head folder
	   This assumes that you have one folder (ex. "My_Data") containing one folder
	   for each strain you are analyzing.
	   My_Data
	      StrainA
	      StrainB
	      ...
	b. Background coordinates
	   These coordinates will be the same for each image analyzed. Assume a specific
	   part of the image will always be background because the worm is designed
	   to be in the middle of the image. Suggested background coordinates for an image
	   of shape (512, 2048) is 0 0 2000 100.
	c. Desired output file name

	Example call: python3 LiveBodyAnalysis.py /mnt/c/Users/Documents/My_Data 0 0 2000 100 output

2. Feature Exploration: FeatureExploration.py User Inputs
	a. Name of file produced by LiveBodyAnalysis.py, which must be located in same folder
	b. Name of ground truth file data, which must be located in same folder
	   File should have one sheet only and a column named 'Name' which denotes the name
	   of the image file exactly as it appears in your folder directory without the .png
	   or .jpg.
	   Name    Feature1    Feature 2    ...
	   Img1    1	       1            ...
	   Img2    0	       1            ...
	   Img3    0	       0            ...
	   ...

	Example call: python3 FeatureExploration.py output.csv gtruth.xlsx

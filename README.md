# Characterising Images on Stack Overflow
## Abstract
Images are increasingly being shared by developers in channels such as question-and-answer forums like Stack Overflow.
Although prior work has pointed out that these images are meaningful and provide complementary information compared to their associated text, the extent to which images are useful for supporting developer tasks such as finding related questions and answers is unknown.
In this paper, we set out to better understand the characteristics of images posted on Stack Overflow.
Through a qualitative analysis, we observe that `undesired output' is the most frequent purpose for sharing images, and that images play an essential role in the questions that contain them.
Towards automated identification of image characteristics, our trained classifiers show that automation is feasible using both text and image related features, e.g., achieving F1-scores of 0.78 and 0.71 for classifying the image sources of desktop application and web browser.
Furthermore, we inject image related features into a state-of-the-art approach for identifying related questions. Our results show that adding image related features improves performance, with higher F1-scores.
Our work presents a first step towards understanding images in Stack Overflow and opens up future avenues for using images to support developer tasks.
## Contents
* `Dataset` - a directory of the dataset that are used in RQ1-3
	* `RQ1` - 
		* `Image_Manual_Classify_768.csv` - all the manual classification of RQ1, i.e., image source, image content, image purposes and so on.
	* `RQ2` -
		* `RQ2_data.xlsx` - the question data that is used in RQ2 with various attributes (question title, body, tag, image link, image source, image content, image purposes).		
		* `Image_text_extract_paddleocr_RQ2.xlsx' - the text that is extracted from the image using paddleocr.
		* `Image_RQ2.zip` - a set of images. 
	* `RQ3` - 
* `Script` - a directory of the scripts that are used in RQ2 (classifier) and RQ3 (question relatedness detection)

		
## Authors
- [Dong Wang](https://dong-w.github.io/) - Nara Institute of Science and Technology
- [Syful Islam]() - Nara Institute of Science and Technology
- [Tao Xiao](https://tao-xiao.github.io/) - Nara Institute of Science and Technology
- [Christoph Treude](https://ctreude.ca/) - The University of Melbourne
- [Raula Gaikovina Kula](https://raux.github.io/) - Nara Institute of Science and Technology
- [Hideaki Hata](https://hideakihata.github.io/) - Shinshu University
- [Zheng Chen]() - Osaka University
- [Kenichi Matsumoto](https://matsumotokenichi.github.io/) - Nara Institute of Science and Technology

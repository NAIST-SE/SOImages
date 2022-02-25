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
* `Dataset` - a directory of the dataset
	* `500_SATD_Comments.csv` - results of RQ1: open coding and card sorting SATD comments in Maven repositories.
		* `Comment Location` - where the SATD comment store at in the repository.
		* `Comment` - the content of the SATD comment.
		* `Keywords` - the set of keywords which are used to extract SATD comment.
		* `Location` - where the SATD comment store at in the build file.
		* `Reason` - why the SATD occure in the build file.
		* `Purpose` - why the developer left the SATD comment in the build file.
## Authors
- [Dong Wang](https://dong-w.github.io/)
- [Syful Islam]()
- [Tao Xiao](https://tao-xiao.github.io/)
- [Christoph Treude](https://ctreude.ca/)
- [Raula Gaikovina Kula](https://raux.github.io/)
- [Hideaki Hata](https://hideakihata.github.io/)
- [Zheng Chen]()
- [Kenichi Matsumoto](https://matsumotokenichi.github.io/)

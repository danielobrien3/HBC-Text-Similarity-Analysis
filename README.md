# Text Similarity Analysis Module.

Multi-tier text analysis tool being built in Python as part of the Marist IBM Joint Study program. This module aims to use multi-tier analysis to subvert the time and resource constraints that are normally tied to text analysis. The software development is done entirely by me, Daniel O'Brien. However, Professor Michael Gildein originally thought of this concept and has been a huge force in providing me direction. 


## Multi-tier Philosophy

The concept of this project is to use multiple tiers of text analysis in order to reduce time wasted by reducing the pool of document's being analyzed. The 'how' is actually quite simple. By breaking documents up into 'tiers', we can get a better understanding of how similar they are without analyzing the entirety of both documents. If we compare the title's of two documents and see little-to-no similarity, we can make the assumption that the documents are not worth comparing any further. We can do the same thing with sub-titles or summary sections. If all of these 'tiers' past our threshold of similarity, only then will we analyze the entirety of both documents. 


## Current Progress

This project has been finished to meet its original requirements. However, this project will be picked back up Spring 2020 to be expanded upon so that the results can be studied. The resulting research will be uploaded to this repo when it is done. 


## Running this locally

To run this locally you must pip install the packages found in the first cell. IMPORTANT: This project uses a word2vec model created by Google. That file was excluded from this repo since it is 3.64GB, but it can be found [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing). It's also important to note that this project will work for any word2vec model you choose to use. 


## Acknowledgments

* My Professor Michael Gildein. 

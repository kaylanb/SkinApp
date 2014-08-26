Must Do steps for app:
1. 1st Functionality of Web App: user is shown image from Test Set (can hit button to go to another image if desired), says whether has People or Food in it, hits submit, learned RandForest predictsfor ALL Test set images, outputs what it thinks was in the specific image user saw AND precision,recall for entire Test set it ran on
    -- problems installing scipy, so will just put Test Set 3 colum output in tmp/ so no 1st butt for now
	-- for user interface have 4 buttons, 1st butt: runs ML on Test set and outputs 3 colm text file: prediction, answer, url, 2nd butt: checks whether that file exists and if so reads it in and outputs results, 3rd butt: calls function that calcs all Blob features and plots them on one subplot saves subplot and subplot is shown to user, 4th butt: same as 3rd but for HOG
2. 2nd Functionality: same as 1st Functionality, but user uploads their own rgb image from local machine and same info entered and returned
3. get working on heroku
4. email sudeep & team link to website for comments and say next steps... 
5. Add "about" html sections for the project, the Blob and HOG methods, intended use, needed/future work, acknowledgements
6. allow for random ordering when constructing TrainX,TestX,TrainY,TestY
    -- for now (8/24/14), for simplicity, I am NOT RANDOM ORDERING TrainX and TestX data sets, instead TrainX is the first half of the Food rows (features) concatenated with the first half of People rows (features), TestX is the same but using second half of rows, and TextY and TrainY are all 0's for Food and all 1's for People concatenated in same way

Upload File code modified from: http://www.runnable.com/
setup(name='Uploading', version='1.0',
      description='Code example demonstrating how to upload a file using Flask',
      author='Miguel Molina', author_email='info@runnable.com',
      url='http://www.runnable.com/')

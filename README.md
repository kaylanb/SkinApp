Must Do steps for app:
1. 1st Functionality of Web App: user is shown image from Test Set (can hit button to go to another image if desired), says whether has People or Food in it, hits submit, learned RandForest predictsfor ALL Test set images, outputs what it thinks was in the specific image user saw AND precision,recall for entire Test set it ran on
2. 2nd Functionality: same as 1st Functionality, but user uploads their own rgb image from local machine and same info entered and returned
3. get working on heroku
4. email sudeep & team link to website for comments and say next steps... 
5. Add "about" html sections for the project, the Blob and HOG methods, intended use, needed/future work, acknowledgements
6. allow for random ordering when constructing TrainX,TestX,TrainY,TestY
    -- for now (8/24/14), for simplicity, I am NOT RANDOM ORDERING TrainX and TestX data sets, instead TrainX is the first half of the Food rows (features) concatenated with the first half of People rows (features), TestX is the same but using second half of rows, and TextY and TrainY are all 0's for Food and all 1's for People concatenated in same way

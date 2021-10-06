'''
How to use

CMD or Powershell
                python.exe mushroom_download_images.py 0 100.000
                python.exe mushroom_download_images.py 100.000 200.000
                python.exe mushroom_download_images.py 200.000 300.000
                python.exe mushroom_download_images.py 300.000 400.000
                python.exe mushroom_download_images.py 400.000 500.000
                python.exe mushroom_download_images.py 500.000 600.000
                python.exe mushroom_download_images.py 600.000 700.000

'''
## Importing Necessary Modules
import requests # to get image from the web
import shutil # to save it locally
import pandas as pd
import sys
from pathlib import Path

df = pd.read_csv(".\dataset\Observations.csv", sep=';', usecols=['image_url'])
min_index = sys.argv[1:][0]
max_index = sys.argv[1:][1]
#print(min_index, max_index)
df = df.loc[int(min_index):int(max_index),:]

#df = df.loc[10000:10003,:]


for index, row in df.iterrows():
    #print row['c1'], row['c2']
    ## Set up the image URL and filename
    #image_url = "https://images.mushroomobserver.org/320/39.jpg" #"https://cdn.pixabay.com/photo/2020/02/06/09/39/summer-4823612_960_720.jpg"
    image_url = row['image_url'] + ".jpg"
    filename = ".\\images\\" + image_url.split("/")[-1]


    if Path(filename).is_file() == False:
        # Open the url image, set stream to True, this will return the stream content.
        r = requests.get(image_url, stream = True)

        # Check if the image was retrieved successfully
        if r.status_code == 200:
            # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
            r.raw.decode_content = True
            
            # Open a local file with wb ( write binary ) permission.
            with open(filename,'wb') as f:
                shutil.copyfileobj(r.raw, f)
                
            print('Image sucessfully Downloaded: ',image_url,filename)
        else:
            print('Image Couldn\'t be retreived :',image_url,filename)


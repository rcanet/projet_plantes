############################
### Eliminate duplicates ###
############################
# Image hashing / perceptual hashing -> Build a "hash value" to uniquely identify an input image based on its content.
# Source : https://pyimagesearch.com/2017/11/27/image-hashing-opencv-python/

# 0 - Import the necessary packages
from imutils import paths
import argparse
import time
import sys
import cv2
import os
import csv
import pandas as pd

# 1 - Definition of the functions
def dhash(image, hashSize=8):
	# resize the input image, adding a single column (width) so we
	# can compute the horizontal gradient
	resized = cv2.resize(image, (hashSize + 1, hashSize))
	
	# compute the (relative) horizontal gradient between adjacent
	# column pixels
	diff = resized[:, 1:] > resized[:, :-1]
	
	# convert the difference image to a hash
	return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


# 2 - Gathering the path where we must find the needles images in the haystack 
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--haystack", required=True,
	help="dataset of images to search through (i.e., the haytack)")
ap.add_argument("-n", "--needles", required=True,
	help="set of images we are searching for (i.e., needles)")
args = vars(ap.parse_args())

haystackPaths = list(paths.list_images(args["haystack"]))
needlePaths = list(paths.list_images(args["needles"]))

# 3 - Grab the base subdirectories for the needle paths, initialize the
# dictionary that will map the image hash to corresponding image,
# hashes, then start the timer
BASE_PATHS = set([p.split(os.path.sep)[-2] for p in needlePaths])
haystack = {}
start = time.time()


# 4 - Extract the images from the haystackPaths
for p in haystackPaths:
	# load the image from disk
	image = cv2.imread(p)
	# if the image is None then we could not load it from disk (so
	# skip it)
	if image is None:
		continue
	# convert the image to grayscale and compute the hash
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	imageHash = dhash(image)
	# update the haystack dictionary
	l = haystack.get(imageHash, [])
	l.append(p)
	haystack[imageHash] = l
	

# 5 - Diagnostic : Time
print("[INFO] processed {} images in {:.2f} seconds".format(
	len(haystack), time.time() - start))
print("[INFO] computing hashes for needles...")


# 6 - Look in the needlePaths
duplicate_groups = 0
total_duplicate_images = 0
list_path_duplicates = [] # Permettra le stockage du chemin des doublons
seen_hashes = set()   # Permet d'éviter d'afficher plusieurs fois le même groups en créeant un ensemble vide
to_remove = [] # Stocker les chemins à supprimer

for p in needlePaths:
    # load the image from disk
    image = cv2.imread(p)
    if image is None:
        continue

    # convert the image to grayscale and compute the hash
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageHash = dhash(image)

    # grab all image paths that match the hash
    matchedPaths = haystack.get(imageHash, [])

    # if we have more than one image with the same hash -> duplicates
    if len(matchedPaths) > 1 and imageHash not in seen_hashes:
        seen_hashes.add(imageHash)
        duplicate_groups += 1
        total_duplicate_images += len(matchedPaths) - 1  # exclude 1 original

        # garder le premier, ajouter les autres à la liste
        to_remove.extend(matchedPaths[1:])

        print(f"\n[DUPLICATES FOUND] {len(matchedPaths)} images partagent le même hash : {imageHash}")
        for mp in matchedPaths:
            list_path_duplicates.append({"hash": imageHash, "path": mp})  

# Supprimer les doublons
for file in to_remove:
    os.remove(file)
    print(f"[REMOVED] {file}")


 # Dataframe with the duplicates paths                 
df_duplicates = pd.DataFrame(list_path_duplicates)
      
pd.DataFrame(df_duplicates).to_csv('../02_data/duplicates_files1.csv', index=None)

print("\n===============================")
print(f"[INFO] Total de groupes de doublons : {duplicate_groups}")
print(f"[INFO] Total d’images doublées (hors originales) : {total_duplicate_images}")
print("===============================")

# Lancer avec : python hash_and_search.py --haystack ../02_data/data_preliminary/ --needles ../02_data/data_preliminary/  
# Il y a au total 370 doublons (sans compter les originales) pour 301 "groupes" de doublons
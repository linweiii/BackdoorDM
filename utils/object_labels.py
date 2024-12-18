import nltk
from nltk.corpus import wordnet as wn
from transformers import ViTForImageClassification

'''
    TODO: automatically get all relevant ImageNet labels based on the course input category
            E.g., 'cat' -> [281, 282, 283, 284, 285, 286, 287]
'''

# # Download WordNet data if not already downloaded
# try:
#     wn.synsets('dog')
# except LookupError:
#     print("Downloading WordNet...")
#     nltk.download('wordnet')
#     print("WordNet downloaded.")


# cls_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
# config = cls_model.config
# imgNet_id2label = config.id2label
# print(imgNet_id2label)

# def print_hyponyms(synset, all_label_names, level=0):
#     hyponyms = synset.hyponyms()

#     if not hyponyms:
#         return
    
#     names = [lemma.name() for lemma in synset.lemmas()] 
#     print(f"{' ' * level}Synset: {synset.name()} - Names: {names}")
#     all_label_names.extend([n.replace('_', ' ') for n in names if n not in all_label_names])
    
#     for hyponym in hyponyms:
#         print_hyponyms(hyponym, all_label_names, level + 2) 

# category_name = 'dog'
# synset = wn.synset(f'{category_name}.n.01')
# all_label_names = []
# print_hyponyms(synset, all_label_names)
# print(all_label_names)
# print()


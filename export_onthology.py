from multiprocessing import Pool
import os

import numpy as np
import json

import pandas as pd

from rdflib import Graph, Namespace, Literal, RDF, RDFS, OWL, BNode, URIRef
from rdflib.namespace import XSD
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map 

if os.path.exists('tag_id_to_name.json'):
    with open('tag_id_to_name.json') as f:
        tag_id_to_name = json.load(f)

ns = "http://www.semanticweb.org/scarlet/ontologies/2024/0/DatabaseOntology#"

DBO = Namespace(ns)
RDF_NS = Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#")

ratingDict = {
    0: "general",
    1: "sensitive",
    2: "questionable",
    3: "explicit"
}

def process_chunk(chunk):
    local_graph = Graph()  # Create a local graph for each chunk
    # Your existing code for `add_post` and other functions
    local_graph.parse('boorudatabase.rdf')
    # Process each row in the chunk
    for _, row in chunk.iterrows():
        add_post(row, local_graph)  # Pass the local graph to add_post

    return local_graph.serialize(format='xml')  # Return the serialized graph

def add_post(csv_row, data_graph, database_id=0):
    postid = csv_row['id']
    createdAt = csv_row['created_at']
    uploaderId = csv_row['uploader_id']
    source = csv_row['source']
    md5 = csv_row['md5']
    parentId = csv_row['parent_id']
    hasChildren = csv_row['has_children']
    isDeleted = csv_row['is_deleted']
    isBanned = csv_row['is_banned']
    pixivId = csv_row['pixiv_id']
    hasActiveChildren = csv_row['has_active_children']
    bitFlags = csv_row['bit_flags']
    hasLarge = csv_row['has_large']
    hasVisibleChildren = csv_row['has_visible_children']
    imageWidth = csv_row['image_width']
    imageHeight = csv_row['image_height']
    fileSize = csv_row['file_size']
    fileExt = csv_row['file_ext']
    ratingString = csv_row['rating']
    score = csv_row['score']
    upScore = csv_row['up_score']
    downScore = csv_row['down_score']
    favCount = csv_row['fav_count']
    fileUrl = csv_row['file_url']
    largeFileUrl = csv_row['large_file_url']
    previewFileUrl = csv_row['preview_file_url']
    if postid != postid:
        return
    post_uri = DBO['Post' + str(postid)]
    data_graph.add((post_uri, RDF.type, DBO['Post']))
    data_graph.add((post_uri, DBO['postId'], Literal(postid, datatype=XSD.integer)))
    data_graph.add((DBO['DanbooruDatabase' + str(database_id)], DBO['hasPost'], post_uri))
    # DanbooruDatabase databaseId 0
    data_graph.add((DBO['DanbooruDatabase' + str(database_id)], DBO['databaseId'], Literal(database_id, datatype=XSD.integer)))
    data_graph.add((post_uri, DBO['hasDatabase'], DBO['DanbooruDatabase' + str(database_id)]))
    data_graph.add((post_uri, DBO['createdAt'], Literal(createdAt, datatype=XSD.dateTime)))
    data_graph.add((post_uri, DBO['uploaderId'], Literal(uploaderId, datatype=XSD.integer)))
    data_graph.add((post_uri, DBO['source'], Literal(source, datatype=XSD.string)))
    data_graph.add((post_uri, DBO['md5'], Literal(md5, datatype=XSD.string)))
    #data_graph.add((post_uri, DBO['parentId'], Literal(int(parentId), datatype=XSD.integer)))
    # -> hasParent (post)
    if parentId:
        parent_uri = DBO['Post' + str(parentId)]
        data_graph.add((post_uri, DBO['hasParent'], parent_uri))
    data_graph.add((post_uri, DBO['hasChildren'], Literal(hasChildren, datatype=XSD.boolean)))
    data_graph.add((post_uri, DBO['isDeleted'], Literal(isDeleted, datatype=XSD.boolean)))
    data_graph.add((post_uri, DBO['isBanned'], Literal(isBanned, datatype=XSD.boolean)))
    # if not NaN
    if pixivId == pixivId: # because NaN != NaN
        data_graph.add((post_uri, DBO['pixivId'], Literal(int(pixivId), datatype=XSD.integer)))
    data_graph.add((post_uri, DBO['hasActiveChildren'], Literal(hasActiveChildren, datatype=XSD.boolean)))
    data_graph.add((post_uri, DBO['bitFlags'], Literal(bitFlags, datatype=XSD.integer)))
    data_graph.add((post_uri, DBO['hasLarge'], Literal(hasLarge, datatype=XSD.boolean)))
    data_graph.add((post_uri, DBO['hasVisibleChildren'], Literal(hasVisibleChildren, datatype=XSD.boolean)))
    data_graph.add((post_uri, DBO['imageWidth'], Literal(imageWidth, datatype=XSD.integer)))
    data_graph.add((post_uri, DBO['imageHeight'], Literal(imageHeight, datatype=XSD.integer)))
    data_graph.add((post_uri, DBO['fileSize'], Literal(fileSize, datatype=XSD.integer)))
    data_graph.add((post_uri, DBO['fileExt'], Literal(fileExt, datatype=XSD.string)))
    data_graph.add((post_uri, DBO['rating'], Literal(ratingDict[ratingString], datatype=XSD.string)))
    data_graph.add((post_uri, DBO['score'], Literal(score, datatype=XSD.integer)))
    data_graph.add((post_uri, DBO['upScore'], Literal(upScore, datatype=XSD.integer)))
    data_graph.add((post_uri, DBO['downScore'], Literal(downScore, datatype=XSD.integer)))
    data_graph.add((post_uri, DBO['favCount'], Literal(favCount, datatype=XSD.integer)))
    data_graph.add((post_uri, DBO['fileUrl'], Literal(fileUrl, datatype=XSD.string)))
    data_graph.add((post_uri, DBO['largeFileUrl'], Literal(largeFileUrl, datatype=XSD.string)))
    data_graph.add((post_uri, DBO['previewFileUrl'], Literal(previewFileUrl, datatype=XSD.string)))
    #data_graph.add((post_uri, DBO['tagList'], Literal(tagList, datatype=XSD.string)))
    def add_tag_with_category(tag_list, tagtype="general"):
        if not isinstance(tag_list, str):
            return
        for tag in tag_list.split():
            # get tag object
            tag_name = tag_id_to_name.get(str(tag))
            if not tag_name:
                continue
            # search in graph for tag
            tag_uri = DBO['Tag' + tag]
            not_exists_tag_in_graph = (tag_uri, RDF.type, None) not in data_graph
            first_letter_capitalized = tagtype[0].upper() + tagtype[1:]
            if not_exists_tag_in_graph:
                data_graph.add((tag_uri, RDF.type, DBO['Tag']))
                data_graph.add((tag_uri, DBO['tagId'], Literal(tag, datatype=XSD.integer)))
                if tagtype:
                    data_graph.add((tag_uri, DBO['tagCategory'], Literal(tagtype, datatype=XSD.string)))
                data_graph.add((tag_uri, DBO['name'], Literal(tag_name, datatype=XSD.string)))
            data_graph.add((post_uri, DBO[f'has{first_letter_capitalized}Tag'], tag_uri))
    for tagtype in ['general', 'artist', 'character', 'copyright', 'meta']:
        add_tag_with_category(csv_row[f'tag_list_{tagtype}'], tagtype)

def main(limit=1000000, multiprocess=True):
    # if limit is true, sample from the dataset
    #df = pd.read_csv('dataset.csv', nrows=limit)
    if limit:
        df = pd.read_csv('dataset.csv', nrows=limit)
    else:
        df = pd.read_csv('dataset.csv')
    print("Loaded dataset")
    # Load the ontology
    ontology_graph = Graph()
    ontology_graph.parse('boorudatabase.rdf')
    # if not multiprocess, just run the code
    if not multiprocess:
        combined_graph = Graph()
        for _, row in tqdm(df.iterrows(), total=df.shape[0]):
            add_post(row, combined_graph)
        with open('database_graph.rdf', 'w', encoding='utf-8') as f:
            f.write(combined_graph.serialize(format='xml'))
        return
    # Assuming 'df' is your DataFrame
    num_processes = 12  # Number of processes to create
    chunk_split_size = 10000
    # Split DataFrame into chunks
    chunks = np.array_split(df, df.shape[0] // chunk_split_size)

    # Create a pool of processes
    results = process_map(process_chunk, chunks, max_workers=num_processes)

    # Combine results
    combined_graph = Graph()
    for result in results:
        combined_graph.parse(data=result, format='xml')

    # Serialize the combined graph
    with open('database_graph.rdf', 'w', encoding='utf-8') as f:
        f.write(combined_graph.serialize(format='xml'))

if __name__ == '__main__':
    def dump_tag_names():
        from db import Tag
        tag_id_to_name = {}
        for tag in Tag.select():
            tag_id_to_name[tag.id] = tag.name
        with open('tag_id_to_name.json', 'w') as f:
            json.dump(tag_id_to_name, f)

    if not os.path.exists('tag_id_to_name.json'):
        dump_tag_names()
    main(limit=None, multiprocess=True)

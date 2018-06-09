import csv
import json
import numpy as np
import scipy.io as sio



def get_object_attributes_str(query):
    """ generate a list of the attributes associated with the objects of a query
    Returns:
    list of tuples: [(u'sky', u'blue'), (u'grass', u'green')]
    """
    n_objects = query.objects.shape[0]
    attributes = []
    
    if not isinstance(query.unary_triples, np.ndarray):
        node = query.unary_triples
        sub_ix = node.subject
        sub_name = query.objects[sub_ix].names
        if isinstance(sub_name, np.ndarray):
            sub_name = sub_name[0]
        tup = (sub_name, node.object)
        attributes.append(tup)
    else:
        n_attributes = query.unary_triples.shape[0]
        for attr_ix in range(0, n_attributes):
            node = query.unary_triples[attr_ix]
            sub_ix = node.subject
            sub_name = query.objects[sub_ix].names
            if isinstance(sub_name, np.ndarray):
                sub_name = sub_name[0]
            tup = (sub_name, node.object)
            attributes.append(tup)
    
    return attributes



def get_partial_scene_matches(images, scenes):
    matches = []
    for q_ix in range(0, len(scenes)):
        scene = scenes[q_ix].annotations
        matching_images = []
        for i_ix in range(0, len(images)):
            image = images[i_ix].annotations
            if does_match(image, scene):
                matching_images.append(i_ix)
        matches.append(matching_images)
    return np.array(matches)



# 519/635 & 0 is a good match (clear glasses on woman)
def does_match(image, scene):
    verbose = False
    
    # are all scene objects in the image?
    scene_objects = []
    for scene_obj_ix in range(0, len(scene.objects)):
        scene_obj_name = scene.objects[scene_obj_ix].names
        scene_objects.append(scene_obj_name)
    
    image_objects = []
    for image_obj_ix in range(0, len(image.objects)):
        image_obj_name = image.objects[image_obj_ix].names
        if isinstance(image_obj_name, np.ndarray):
            image_obj_name = image_obj_name[0]
        image_objects.append(image_obj_name)
    
    is_subset = set(scene_objects).issubset(set(image_objects))
    if not is_subset:
        if verbose: print '{} not ss of {}'.format(scene_objects, image_objects)
        return False
    
    # are all scene object attributes in the image?
    scene_oa = get_object_attributes_str(scene)
    image_oa = get_object_attributes_str(image)
    is_subset = set(scene_oa).issubset(set(image_oa))
    if not is_subset:
        if verbose: print '{} not ss of {}'.format(scene_oa, image_oa)
        return False
  
    # are all scene relationships in the image?
    scene_triples = []
    if isinstance(scene.binary_triples, np.ndarray):
        for trip in scene.binary_triples:
            scene_triples.append(trip)
    else:
        scene_triples.append(scene.binary_triples)
    
    image_triples = []
    if isinstance(image.binary_triples, np.ndarray):
        for trip in image.binary_triples:
            image_triples.append(trip)
    else:
        image_triples.append(image.binary_triples)
    
    scene_rels = []
    for scene_trip in scene_triples:
        scene_sub_ix = scene_trip.subject
        scene_sub_name = scene.objects[scene_sub_ix].names
        
        scene_obj_ix = scene_trip.object
        scene_obj_name = scene.objects[scene_obj_ix].names
        
        rel_str = '{} {} {}'.format(scene_sub_name, scene_trip.predicate, scene_obj_name)
        scene_rels.append(rel_str)
    
    image_rels = []
    for image_trip in image_triples:
        rel_str = '{} {} {}'.format(image_trip.text[0], image_trip.text[1], image_trip.text[2])
        image_rels.append(rel_str)
    
    is_subset = set(scene_rels).issubset(set(image_rels))
    if not is_subset:
        if verbose: print '{} not ss of {}'.format(scene_rels, image_rels)
    return is_subset



#===============================================================================
#
#
def get_data(data_path):
    print("loading vg_data file...")
    vgd_path = data_path + "vg_data.mat"
    vgd = sio.loadmat(vgd_path, struct_as_record=False, squeeze_me=True)
        
    print("loading test queries...")
    query_path = data_path + "simple_graphs.mat"
    queries = sio.loadmat(query_path, struct_as_record=False, squeeze_me=True)
    
    matches = get_partial_scene_matches(vgd['vg_data_test'], queries['simple_graphs'])
    return matches



def read_match_file(match_file):
    f = open(match_file, 'rb')
    csv_r = csv.reader(f)
    matches = []
    for row in csv_r:
        matches.append(map(int, row))
    return matches



#===============================================================================
# MAIN
#
if __name__ == '__main__':
    matches = None
    if match_file is None:
        cfg_file = open('config.json')
        cfg_data = json.load(cfg_file)
        
        mat_path = cfg_data['file_paths']['mat_path']
        matches = get_data(mat_path)
        
        data_path = cfg_data['file_paths']['data_path']
        match_file = os.path.join(data_path, 'partial_query_matches.csv')
        with open(match_file, 'wb') as f:
            csv_w = csv.writer(f)
            for q in matches:
                csv_w.writerow(q)
    else:
        matches = read_match_file(match_file)
    
    import pdb; pdb.set_trace()
    n_queries = len(matches)
    for query_ixs in range(0, n_queries):
        ratk = get_ratk(irsg_energies[query_ix], matches[query_ix])
        irsg_ratk.append(ratk)
    print 'done!'

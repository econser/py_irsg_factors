#from __future__ import print_function

import os
import json
import numpy as np
import opengm as ogm
import scipy.io as sio



#===============================================================================
# UTILS
#
def get_object_detections(image_ix, potentials_mat, platt_mod):
    """Get object detection data from an image
    Input:
        image_ix: image number
        potentials_mat: potentials .mat file
        platt_mod_mat: platt model .mat file
    Output:
        dict: object name (str) -> boxes (numpy array of [x,y,w,h,p] entries), platt model applied to probabilites
    """
    object_mask = [name[:3] == 'obj' for name in potentials_mat['potentials_s'].classes]
    object_mask = np.array(object_mask)
    object_names = potentials_mat['potentials_s'].classes[object_mask]
    object_detections = get_class_detections(image_ix, potentials_mat, platt_mod, object_names)
    return object_detections



def get_attribute_detections(image_ix, potentials_mat, platt_mod):
    """Get object detection data from an image
    Input:
        image_ix: image number
        potentials_mat: potentials .mat file
        platt_mod_mat: platt model .mat file
    Output:
        dict: attribute name (str) -> boxes (numpy array of [x,y,w,h,p] entries), platt model applied to probabilites
    """
    attr_mask = [name[:3] == 'atr' for name in potentials_mat['potentials_s'].classes]
    attr_mask = np.array(attr_mask)
    attr_names = potentials_mat['potentials_s'].classes[attr_mask]
    attr_detections = get_class_detections(image_ix, potentials_mat, platt_mod, attr_names)
    return attr_detections



def get_class_detections(image_ix, potential_data, platt_mod, object_names, verbose=False):
    """Generate box & score values for an image and set of object names
    
    Args:
      image_ix (int): the image to generate detections from
      potential_data (.mat data): potential data (holds boxes, scores, and class to index map)
      platt_data (.mat data): holds platt model parameters
      object_names (numpy array of str): the names of the objects to detect
      verbose (bool): default 'False'
    
    Returns:
      dict: object name (str) -> boxes (numpy array)
    """
    n_objects = object_names.shape[0]
    detections = np.empty(n_objects, dtype=np.ndarray)
    
    box_coords = np.copy(potential_data['potentials_s'].boxes[image_ix])
    box_coords[:,2] -= box_coords[:,0]
    box_coords[:,3] -= box_coords[:,1]
    
    class_to_index_keys = potential_data['potentials_s'].class_to_idx.serialization.keys
    class_to_index_vals = potential_data['potentials_s'].class_to_idx.serialization.values
    obj_id_dict = dict(zip(class_to_index_keys, class_to_index_vals))
    
    det_ix = 0
    for o in object_names:
        if o not in obj_id_dict:
            continue
        
        obj_ix = obj_id_dict[o]
        obj_ix -= 1 # matlab is 1-based
        
        a = 1.0
        b = 1.0
        platt_keys = platt_mod['platt_models'].s_models.serialization.keys
        platt_vals = platt_mod['platt_models'].s_models.serialization.values
        platt_dict = dict(zip(platt_keys, platt_vals))
        if o in platt_dict:
            platt_coeff = platt_dict[o]
            a = platt_coeff[0]
            b = platt_coeff[1]
        
        scores = potential_data['potentials_s'].scores[image_ix][:,obj_ix]
        scores = 1.0 / (1.0 + np.exp(a * scores + b))
        
        n_detections = scores.shape[0]
        scores = scores.reshape(n_detections, 1)
        
        class_det = np.concatenate((box_coords, scores), axis=1)
        detections[det_ix] = class_det
        if verbose: print "%d: %s" % (det_ix, o)
        det_ix += 1
        
    return dict(zip(object_names, detections))



def get_object_attributes(query):
    """ generate a list of the attributes associated with the objects of a query
    
    The unary_triples field has a subject and object:
    blue    sky   above   green   tree   (query ix 109)
    U1,S0   O1    B1      U2,S1   O2
    subject - index of the object to which the attribute applies (1)
    object - the attribute name (blue)
    attributes 1 1 : 'blue'
               2 1 : 'green'
    
    Args:
        query (.mat file): an entry in the list of queries from a .mat file
    
    Returns:
        numpy array: [(0, u'blue'), (1, u'green')]
    """
    n_objects = query.objects.shape[0]
    attributes = []
    
    if not isinstance(query.unary_triples, np.ndarray):
        node = query.unary_triples
        tup = (node.subject, node.object)
        attributes.append(tup)
    else:
        n_attributes = query.unary_triples.shape[0]
        for attr_ix in range(0, n_attributes):
            node = query.unary_triples[attr_ix]
            tup = (node.subject, node.object)
            attributes.append(tup)
    
    return np.array(attributes)



#===============================================================================
# DATA PULL CALLS
#
class RelationshipParameters (object):
    def __init__(self, platt_a, platt_b, gmm_weights, gmm_mu, gmm_sigma):
        self.platt_a = platt_a
        self.platt_b = platt_b
        self.gmm_weights = gmm_weights
        self.gmm_mu = gmm_mu
        self.gmm_sigma = gmm_sigma



class ImageFetchDataset (object):
    def __init__(self, vg_data, potentials_data, platt_models, relationship_models, base_image_path):
        self.vg_data = vg_data
        self.potentials_data = potentials_data
        self.platt_models = platt_models
        self.relationship_models = relationship_models
        self.base_image_path = base_image_path
        
        self.current_image_num = -1
        self.object_detections = None
        self.attribute_detections = None
        self.per_object_attributes = None
        self.image_filename = ""
        self.current_sg_query = None
    
    def configure(self, test_image_num, sg_query):
        if test_image_num != self.current_image_num:
            self.current_image_num = test_image_num
            self.object_detections = get_object_detections(self.current_image_num, self.potentials_data, self.platt_models)
            self.attribute_detections = get_attribute_detections(self.current_image_num, self.potentials_data, self.platt_models)
            self.image_filename = self.base_image_path + os.path.basename(self.vg_data[self.current_image_num].image_path)
            
            if sg_query != self.current_sg_query:
                self.current_sg_query = sg_query
                self.per_object_attributes = get_object_attributes(self.current_sg_query)



def get_mat_data(data_path):
    """
    load the data files for use in running the model.
    expects the files to be all in the same directory
    
    Args:
    data_path: the fully-quaified path to the data files
    
    Returns:
    vgd (.mat file): vg_data file
    potentials (.mat file): potentials file
    platt_mod (.mat file): platt model file
    bin_mod (.mat file): GMM parameters
    queries (.mat file): queries
    """
    print("loading vg_data file...")
    vgd_path = data_path + "vg_data.mat"
    vgd = sio.loadmat(vgd_path, struct_as_record=False, squeeze_me=True)
    
    print("loading potentials data...")
    potentials_path = data_path + "potentials_s.mat"
    potentials = sio.loadmat(potentials_path, struct_as_record=False, squeeze_me=True)
    
    print("loading binary model data...")
    binary_path = data_path + "binary_models_struct.mat"
    bin_mod_mat = sio.loadmat(binary_path, struct_as_record=False, squeeze_me=True)
    bin_mod = get_relationship_models(bin_mod_mat)
    
    print("loading platt model data...")
    platt_path = data_path + "platt_models_struct.mat"
    platt_mod = sio.loadmat(platt_path, struct_as_record=False, squeeze_me=True)
    
    print("loading test queries...")
    query_path = data_path + "simple_graphs.mat"
    queries = sio.loadmat(query_path, struct_as_record=False, squeeze_me=True)
    
    return vgd, potentials, platt_mod, bin_mod, queries



def get_relationship_models(binary_model_mat):
    """Convert the mat file binary model storage to a more convienent structure for python
    Input:
      mat bin model file from sio
        keys
        values
        gmm_params
        platt_params
    Output:
      map from string to relationship_parameters
        'man' -> rel_params
    """
    # create a map from trip_string -> index (e.g. 'shirt_on_man' -> 23)
    trip_ix_root = binary_model_mat['binary_models_struct'].s_triple_str_to_idx.serialization
    trip_to_index_keys = trip_ix_root.keys
    trip_to_index_vals = trip_ix_root.values
    trip_str_dict = dict(zip(trip_to_index_keys, trip_to_index_vals))
    
    # for each trip_str key, pull params from the mat and generate a RelationshipParameters object
    param_list = []
    for trip_str in trip_to_index_keys:
        ix = trip_str_dict[trip_str]
        ix -= 1 # MATLAB uses 1-based indexing here
        platt_params = binary_model_mat['binary_models_struct'].platt_models[ix]
        gmm_params = binary_model_mat['binary_models_struct'].models[ix].gmm_params
        rel_params = RelationshipParameters(platt_params[0], platt_params[1], gmm_params.ComponentProportion, gmm_params.mu, gmm_params.Sigma.T)
        param_list.append(rel_params)
    
    str_to_param_map = dict(zip(trip_to_index_keys, param_list))
    return str_to_param_map



def get_all_data():
    cfg_file = open('config.json')
    cfg_data = json.load(cfg_file)
    
    out_path = cfg_data['file_paths']['output_path']
    img_path = cfg_data['file_paths']['image_path']
    mat_path = cfg_data['file_paths']['mat_path']
    
    vgd, potentials, platt_mod, bin_mod, queries = get_mat_data(mat_path)
    ifdata = ImageFetchDataset(vgd['vg_data_test'], potentials, platt_mod, bin_mod, img_path)
    
    return vgd, potentials, platt_mod, bin_mod, queries, ifdata



#===============================================================================
# GEOMETRIC MEAN ENERGY
#
def get_geomean_scores(image_ixs, query_obj, query_ix, potentials, platt_models, output_path=''):
    import cPickle
    
    object_names = []
    for obj in query_obj.objects:
        object_names.append(obj.names)
    
    image_scores = []
    viz_data = []
    top_box_data = []
    
    for image_ix in image_ixs:
        object_probs = []
        bbox_list = []
        top_list = []
        obj_detections = get_object_detections(image_ix, potentials, platt_models)
        for obj_name in object_names:
            obj_key = 'obj:' + obj_name
            if not obj_detections.has_key(obj_key):
                import pdb; pdb.set_trace()
                continue
            object_probs.append(obj_detections[obj_key][:,4])
            
            sort_ixs = np.argsort(obj_detections[obj_key][:,4])
            
            best_bbox_ix = sort_ixs[::-1][0]
            best_bbox = obj_detections[obj_key][best_bbox_ix, 0:4]
            bbox_tup = (obj_name, best_bbox)
            bbox_list.append(bbox_tup)
            
            top_n = sort_ixs[::-1][:10]
            top_bboxes = obj_detections[obj_key][top_n, 0:4]
            top_scores = obj_detections[obj_key][top_n, 4]
            top_tup = (obj_name, top_bboxes, top_scores)
            top_list.append(top_tup)
        object_probs = np.array(object_probs)
        viz_data.append(bbox_list)
        top_box_data.append(top_list)
        
        top_object_ixs = np.argmax(object_probs, axis=1)
        top_probs = []
        for i, probs in enumerate(object_probs):
            top_probs.append(probs[top_object_ixs[i]])
        top_probs = np.array(top_probs)
        gmean = top_probs.prod()**(1.0/len(top_probs))
        image_scores.append((image_ix, np.exp(-gmean)))
    
    if output_path != '':
        filename = 'q{:03d}_energy_values.csv'.format(query_ix)
        fq_filename = '{}{}'.format(output_path, filename)
        np.savetxt(fq_filename, image_scores, delimiter=',', header='image_ix, energy', fmt='%d, %3.4f', comments='')
        
        filename = 'q{:03d}_best_bboxes.csv'.format(query_ix)
        fq_filename = '{}{}'.format(output_path, filename)
        f = open(fq_filename, 'wb')
        cPickle.dump(viz_data, f, cPickle.HIGHEST_PROTOCOL)
        f.close()
        
        filename = 'q{:03d}_top10_bboxes.csv'.format(query_ix)
        fq_filename = '{}{}'.format(output_path, filename)
        f = open(fq_filename, 'wb')
        cPickle.dump(top_box_data, f, cPickle.HIGHEST_PROTOCOL)
        f.close()
    
    return np.array(image_scores), viz_data, top_box_data



#===============================================================================
# FACTOR-BASED ENERGY
#
def generate_pgm(if_data, verbose=False):
    # gather data from the if data object
    query_graph = if_data.current_sg_query
    object_detections = if_data.object_detections
    attribute_detections = if_data.attribute_detections
    relationship_models = if_data.relationship_models
    per_object_attributes = if_data.per_object_attributes
    image_filename = if_data.image_filename
    
    # generate the graphical model (vg_data_build_gm_for_image)
    n_objects = len(query_graph.objects)
    n_vars = []
    object_is_detected = []
    query_to_pgm = []
    pgm_to_query = []
    
    master_box_coords = []
    
    varcount = 0
    for obj_ix in range(0, n_objects):
        query_object_name = query_graph.objects[obj_ix].names
        
        # occasionally, there are multiple object names (is 0 the best?)
        if isinstance(query_object_name, np.ndarray):
          query_object_name = query_object_name[0]
          
        object_name = "obj:" + query_object_name
        if object_name not in object_detections:
            object_is_detected.append(False)
            query_to_pgm.append(-1)
        else:
            if len(master_box_coords) == 0:
                master_box_coords = np.copy(object_detections[object_name][:,0:4])
            object_is_detected.append(True)
            query_to_pgm.append(varcount)
            varcount += 1
            pgm_to_query.append(obj_ix)
            
            n_labels = len(object_detections[object_name])
            n_vars.append(n_labels)
    
    gm = ogm.gm(n_vars, operator='adder')
    if verbose:
        print "number of variables: {0}".format(gm.numberOfVariables)
        for l in range(0, gm.numberOfVariables):
            print "  labels for var {0}: {1}".format(l, gm.numberOfLabels(l))
    
    functions = []
    
    # generate 1st order functions for objects
    # TODO: test an uniform dist for missing objects
    if verbose: print "unary functions - objects:"
    
    unary_dets = []
    is_cnn_detected = []
    for obj_ix in range(0, n_objects):
        fid = None
        
        pgm_ix = query_to_pgm[obj_ix]
        object_name = query_graph.objects[obj_ix].names
        if isinstance(object_name, np.ndarray):
            object_name = object_name[0]
        detail = "unary function for object '{0}'".format(object_name)
        
        if object_is_detected[obj_ix]:
            if verbose: print "  adding {0} as full explicit function (qry_ix:{1}, pgm_ix:{2})".format(detail, obj_ix, pgm_ix)
            is_cnn_detected.append(True)
            prefix_object_name = "obj:" + object_name
            detections = object_detections[prefix_object_name]
            unary_dets.append(detections[:,4])
            log_scores = -np.log(detections[:,4])
            fid = gm.addFunction(log_scores)
        else:
            if verbose: print "  skipping {0}, no detection available (qry_ix:{1})".format(object_name, obj_ix)
            continue
        
        func_detail = FuncDetail(fid, [pgm_ix], "explicit", "object unaries", detail)
        functions.append(func_detail)
    
    #generate 1st order functions for attributes
    if verbose: print "unary functions - attributes:"
    n_attributes = len(per_object_attributes)
    for attr_ix in range(0, n_attributes):
        obj_ix = int(per_object_attributes[attr_ix][0])
        pgm_ix = query_to_pgm[obj_ix]
        attribute_name = per_object_attributes[attr_ix][1]
        prefix_attribute_name = "atr:" + attribute_name
        
        if prefix_attribute_name not in attribute_detections:
            if verbose: print "  skipping attribute '{0}' for object '{1}' (qry_ix:{2}), no attribute detection available".format(attribute_name, query_graph.objects[obj_ix].names, obj_ix)
            continue
        
        if not object_is_detected[obj_ix]:
            if verbose: print "  skipping attribute '{0}' for object '{1}' (qry_ix:{2}), no object detection available".format(attribute_name, query_graph.objects[obj_ix].names, obj_ix)
            continue
        
        detections = attribute_detections[prefix_attribute_name]
        log_scores = -np.log(detections[:,4])
        
        detail = "unary function for attribute '{0}' of object '{1}' (qry_ix:{2}, pgm_ix:{3})".format(attribute_name, query_graph.objects[obj_ix].names, obj_ix, pgm_ix)
        if verbose: print "  adding {0}".format(detail)
        
        fid = gm.addFunction(log_scores)
        func_detail = FuncDetail(fid, [pgm_ix], "explicit", "attribute unaries", detail)
        functions.append(func_detail)
    
    # generate a tracker for storing obj/attr/rel likelihoods (pre-inference)
    tracker = DetectionTracker(image_filename)
    for i in range(0, n_objects):
        if object_is_detected[i]:
            if isinstance(query_graph.objects[i].names, np.ndarray):
                tracker.object_names.append(query_graph.objects[i].names[0])
            else:
                tracker.object_names.append(query_graph.objects[i].names)
    tracker.unary_detections = unary_dets
    tracker.box_coords = master_box_coords
    tracker.detected_vars = is_cnn_detected
    
    # generate 2nd order functions for binary relationships
    trip_root = query_graph.binary_triples
    trip_list = []
    if isinstance(trip_root, sio.matlab.mio5_params.mat_struct):
        trip_list.append(query_graph.binary_triples)
    else:
        # if there's only one relationship, we don't have an array :/
        for trip in trip_root:
            trip_list.append(trip)
    
    # generate a single cartesian product of the boxes
    # this will only work when all objects are detected across the same boxes
    # we know this is the case for this implementation
    master_cart_prod = None
    for i in range(0, n_objects):
        if object_is_detected[i]:
            obj_name = query_graph.objects[i].names
            boxes = None
            if isinstance(obj_name, np.ndarray):
                boxes = object_detections["obj:"+obj_name[0]]
            else:
                boxes = object_detections["obj:"+obj_name]
            master_cart_prod = np.array([x for x in itertools.product(boxes, boxes)])
            break
    tracker.box_pairs = master_cart_prod
    
    # process each binary triple in the list
    if verbose: print "binary functions:"
    for trip in trip_list:
        sub_ix = trip.subject
        sub_pgm_ix = query_to_pgm[sub_ix]
        subject_name = query_graph.objects[sub_ix].names
        if isinstance(subject_name, np.ndarray):
            subject_name = subject_name[0]
        
        obj_ix = trip.object
        obj_pgm_ix = query_to_pgm[obj_ix]
        object_name = query_graph.objects[obj_ix].names
        if isinstance(object_name, np.ndarray):
            object_name = object_name[0]
        
        relationship = trip.predicate
        bin_trip_key = subject_name + "_" + relationship.replace(" ", "_")  + "_" + object_name
        
        # check if there is a gmm for the specific triple string
        if bin_trip_key not in relationship_models:
            if verbose: print "  no model for '{0}', generating generic relationship string".format(bin_trip_key)
            bin_trip_key = "*_" + relationship.replace(" ", "_") + "_*"
            if bin_trip_key not in relationship_models:
                if verbose: print "    skipping binary function for relationship '{0}', no model available".format(bin_trip_key)
                continue
        
        # verify object detections
        if sub_ix == obj_ix:
            if verbose: print "    self-relationships not possible in OpenGM, skipping relationship"
            continue
        
        if not object_is_detected[sub_ix]:
            if verbose: print "    no detections for object '{0}', skipping relationship (qry_ix:{1})".format(subject_name, sub_ix)
            continue
        
        if not object_is_detected[obj_ix]:
            if verbose: print "    no detections for object '{0}', skipping relationship (qry_ix:{1})".format(object_name, obj_ix)
            continue
        
        # get model parameters
        prefix_object_name = "obj:" + object_name
        bin_object_box = object_detections[prefix_object_name]
        
        prefix_subject_name = "obj:" + subject_name
        bin_subject_box = object_detections[prefix_subject_name]
        
        rel_params = relationship_models[bin_trip_key]
        
        # generate features from subject and object detection boxes
        cart_prod = master_cart_prod
        sub_dim = 0
        obj_dim = 1
        
        subx_center = cart_prod[:, sub_dim, 0] + 0.5 * cart_prod[:, sub_dim, 2]
        suby_center = cart_prod[:, sub_dim, 1] + 0.5 * cart_prod[:, sub_dim, 3]
        
        objx_center = cart_prod[:, obj_dim, 0] + 0.5 * cart_prod[:, obj_dim, 2]
        objy_center = cart_prod[:, obj_dim, 1] + 0.5 * cart_prod[:, obj_dim, 3]
        
        sub_width = cart_prod[:, sub_dim, 2]
        relx_center = (subx_center - objx_center) / sub_width
        
        sub_height = cart_prod[:, sub_dim, 3]
        rely_center = (suby_center - objy_center) / sub_height
        
        rel_height = cart_prod[:, obj_dim, 2] / cart_prod[:, sub_dim, 2]
        rel_width = cart_prod[:, obj_dim, 3] / cart_prod[:, sub_dim, 3]
        
        features = np.vstack((relx_center, rely_center, rel_height, rel_width)).T
        
        #tracker.box_pairs = np.copy(cart_prod) #TODO: is this copy necessary?
        #tracker.box_pairs = cart_prod
        
        # generate scores => log(epsilon+scores) => platt sigmoid
        scores = gmm_pdf(features, rel_params.gmm_weights, rel_params.gmm_mu, rel_params.gmm_sigma)
        eps = np.finfo(np.float).eps
        scores = np.log(eps + scores)
        sig_scores = 1.0 / (1. + np.exp(rel_params.platt_a * scores + rel_params.platt_b))
        
        log_likelihoods = -np.log(sig_scores)
        
        #tracker.add_group(bin_trip_key, np.copy(log_likelihoods), np.copy(bin_object_box), object_name, np.copy(bin_subject_box), subject_name) # TODO: are these copy calls necessary?
        tracker.add_group(bin_trip_key, log_likelihoods, bin_object_box, object_name, bin_subject_box, subject_name)
        
        # generate the matrix of functions
        n_subject_val = len(bin_subject_box)
        n_object_val = len(bin_object_box)
        bin_functions = np.reshape(log_likelihoods, (n_subject_val, n_object_val)) # TODO: determine if any transpose is needed
        if obj_pgm_ix < sub_pgm_ix: bin_functions = bin_functions.T
        
        # add binary functions to the GM
        detail = "binary functions for relationship '%s'" % (bin_trip_key)
        if verbose: print("    adding %s" % detail)
        fid = gm.addFunction(bin_functions)
        
        var_indices = [sub_pgm_ix, obj_pgm_ix]
        if obj_pgm_ix < sub_pgm_ix: var_indices = [obj_pgm_ix, sub_pgm_ix]
        func_detail = FuncDetail(fid, var_indices, "explicit", "binary functions", detail)
        functions.append(func_detail)
        
    # add 1st order factors (fid)
    for f in functions:
        n_indices = len(f.var_indices)
        if n_indices == 1:
            if verbose:
                print "  adding unary factor: {0}".format(f.detail)
                print "    fid: {0}   var: {1}".format(f.gm_function_id.getFunctionIndex(), f.var_indices[0])
            gm.addFactor(f.gm_function_id, f.var_indices[0])
        elif n_indices == 2:
            if verbose:
                print "  adding binary factor: {0}".format(f.detail)
                print "    fid: {0}   var: {1}".format(f.gm_function_id.getFunctionIndex(), f.var_indices)
            gm.addFactor(f.gm_function_id, f.var_indices)
        else:
            if verbose: print "skipping unexpected factor with {0} indices: {1}".format(n_indices, f.function_type)
    
    return gm, tracker



def do_inference(gm, n_steps=120, damping=0., convergence_bound=0.001, verbose=False):
    """ Run belief propagation on the providede graphical model
    returns:
      energy (float): the energy of the GM
      var_indices (numpy array): indices for the best label for each variable
    """
    ogm_params = ogm.InfParam(steps=n_steps, damping=damping, convergenceBound=convergence_bound)
    infr_output = ogm.inference.BeliefPropagation(gm, parameter=ogm_params)
    
    if verbose:
        infr_output.infer(infr_output.verboseVisitor())
    else:
        infr_output.infer()
    
    detected_vars = []
    for i in range(0, gm.numberOfVariables):
        if gm.numberOfLabels(i) > 1:
            detected_vars.append(i)
    
    infr_marginals = infr_output.marginals(detected_vars)
    infr_marginals = np.exp(-infr_marginals)
    
    infr_best_match = infr_output.arg()
    infr_energy = infr_output.value()
    
    return infr_energy, infr_best_match, infr_marginals



def gmm_pdf(X, mixture, mu, sigma):
    n_components = len(mixture)
    n_vals = len(X)
    
    mixed_pdf = np.zeros(n_vals)
    for i in range(0, n_components):
       mixed_pdf += mvn.pdf(X, mu[i], sigma[i]) * mixture[i]
    
    return mixed_pdf



#===============================================================================
# MAIN
#
if __name__ == '__main__':
    vgd, potentials, platt_mod, bin_mod, queries, ifdata = get_all_data()
    import pdb; pdb.set_trace()
    
    query_ix = 0
    img_ix = 0
    
    query = queries['simple_graphs'][query_ix].annotations
    ifdata.configure(img_ix, query)
    gm, tracker = generate_pgm(ifdata)
    energy, best_bboxes, var_marginals = do_inference(gm)    
    
    import pdb; pdb.set_trace()
    print 'done'

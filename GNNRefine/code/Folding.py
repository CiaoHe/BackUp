#!/usr/bin/env python3
# encoding: utf-8

import os, pickle
import random, time, tempfile
import numpy as np
from pyrosetta import *
import Utils as Utils
import multiprocessing

def gen_pot_from_pred(seq, pred_data, ):
    '''
    Generate the potential from prediction.
    '''
    CONFIG = {
        'ALPHA': 1.57,
        'MEFF': 0.0001, 
        'DIST_START': 4.25, # the start distance
        'DIST_START_INDEX': 5,  # the index of the bin corresponding to the start distance
        'DIST_CUT': 19.5,  # the distance cutoff as background 
        'DIST_STEP': 0.5,  # the bin width
        'DIST_REP': [0.0, 2.0, 3.5],  # the x values for repulsion
        'DIST_POT_REP': [10.0, 3.0, 0.5],  # the y values for repulsion
        'DIST_POT_BASE': -0.5,  # the base energy
    }
    
    potentials = {'distCB': {}}
    dist = pred_data['distCB']
    bin_num = dist[0].shape[-1] - CONFIG['DIST_START_INDEX']
    bins = np.array([CONFIG['DIST_START']+CONFIG['DIST_STEP']*i for i in range(bin_num)])
    prob = np.sum(dist[:, CONFIG['DIST_START_INDEX']:], axis=-1)
    bkgr = np.array((bins/CONFIG['DIST_CUT'])**CONFIG['ALPHA'])
    attr = -np.log((dist[:, CONFIG['DIST_START_INDEX']:]+CONFIG['MEFF'])/(dist[:,-1][:,None]*bkgr[None,:]))+CONFIG['DIST_POT_BASE']
    repul = np.maximum(attr[:,0], np.zeros((attr.shape[0])))[:,None]+np.array(CONFIG['DIST_POT_REP'])[None,:]
    dist = np.concatenate([repul, attr], axis=-1)
    bins = np.concatenate([CONFIG['DIST_REP'], bins])
    # x_axis
    potentials['distCB']['x_axis'] = bins
    # y_axis
    for _i, _pair in enumerate(pred_data['adj_pair']):
        a,b,p = _pair[0], _pair[1], prob[_i]
        if b>a:
            a_atom = 'CA' if seq[a]=='G' else 'CB'
            b_atom = 'CA' if seq[b]=='G' else 'CB'
            atom_pair = "%d-%d"%(a, b)
            atom_types = "%s-%s"%(a_atom, b_atom)
            potentials['distCB'][atom_pair] = (atom_types, p, dist[_i])
    
    print("distCB restraints: %d"%(len(potentials['distCB'])-1))
    return potentials

def gen_RosettaRst_from_pot(potentials, rst_dir, rst_type='distCB', dist_step=0.5):
    '''
    Generate the rst files for rosetta.
    '''
    if not os.path.isdir(rst_dir): os.mkdir(rst_dir)
    rosetta_rsts = {}
    rosetta_rsts[rst_type] = []
    x_axis = potentials[rst_type]['x_axis']
    for pair in potentials[rst_type]:
        if pair=="x_axis": continue
        a, b = [int(_) for _ in pair.split('-')]
        a_atom, b_atom = [_ for _ in potentials[rst_type][pair][0].split('-')]
        prob = potentials[rst_type][pair][1]
        y_axis = potentials[rst_type][pair][2]
        assert len(x_axis)==len(y_axis), "the sizes of x_axis and y_axis are not match."
        rst_file = rst_dir+"/%s-%d-%d.txt"%(rst_type, a+1, b+1)
        with open(rst_file, "w") as f:
            f.write('x_axis'+'\t%.3f'*len(x_axis)%tuple(x_axis)+'\n')
            f.write('y_axis'+'\t%.3f'*len(x_axis)%tuple(y_axis)+'\n')

        rst_line = 'AtomPair %s %d %s %d SPLINE %s %s 1.0 %.3f %.5f'%(a_atom, a+1, b_atom, b+1, rst_type, rst_file, 1.0, dist_step)
        rosetta_rsts[rst_type].append([a, b, prob, rst_line])
    
    return rosetta_rsts


def apply_rst(pose, rst, sep1, sep2, pcut, refined_pdb, seq, nogly=False):
    array = [line for a,b,p,line in rst['distCB'] if abs(a-b)>=sep1 and abs(a-b)<sep2 and p>=pcut]
    if len(array) < 1: return
    random.shuffle(array)

    # save rst to file
    # print("rst num:", len(array))
    rst_file = refined_pdb+".%s.rst"%(int(time.time()))
    with open(rst_file,'w') as f: f.write('\n'.join(array)+'\n')

    # add rst on pose
    # print(rst_file)
    constraints = rosetta.protocols.constraint_movers.ConstraintSetMover()
    constraints.constraint_file(rst_file)
    constraints.add_constraints(True)
    constraints.apply(pose)

    os.remove(rst_file)

def folding_by_FastRelax(start_pdb, rst_infos, refined_pdb, rst_weight=2, PCUT=0.15, max_iter=500, ):
    print(start_pdb, refined_pdb)
    # init PyRosetta
    init('-hb_cen_soft -relax:default_repeats 5 -default_max_cycles 200 -out:level 100')

    # initialize pose
    seq = Utils.get_seqs_from_pdb(start_pdb)
    pose = pose_from_pdb(start_pdb)
    # full-atom score_function
    sf_fa = create_score_function('ref2015')
    sf_fa.set_weight(rosetta.core.scoring.atom_pair_constraint, rst_weight)
    switch = SwitchResidueTypeSetMover("fa_standard")
    switch.apply(pose)
    # MoveMap
    mmap = MoveMap()
    mmap.set_bb(True)
    mmap.set_chi(True)
    mmap.set_jump(True)
    # FastRelax
    relax = rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(sf_fa)
    relax.max_iter(max_iter)
    relax.dualspace(True)
    relax.set_movemap(mmap)    
    # appy the rst to pose
    apply_rst(pose, rst_infos, 1, len(seq), PCUT, refined_pdb, seq, nogly=False)
    # relax
    print('folding...')
    relax.apply(pose)
    refined_energy = sf_fa(pose)
    # save model
    pose.dump_pdb(refined_pdb)
    print(refined_pdb, 'done.')
    return start_pdb, refined_pdb, refined_energy

def folding(pred_file, work_dir, n_decoy=1, n_proc=1):
    """
    Fold pdb based on the predicted distance probability.
    Args:
        pred_file (str): the pkl file of predicted distance probability.
        work_dir (str): work dir.
        n_decoy (int): number of decoy models built in each iteration.
        n_proc (int): number of processes running in parallel (>=1).
    """
    print(pred_file, n_decoy)
    process_results = []
    assert n_proc>=1, 'n_proc must be >=1.'
    process_pool = multiprocessing.Pool(processes=n_proc)
    predictions = pickle.load(open(pred_file, "rb"))
    for item in predictions:
        start_pdb = item['pdb']

        rst_dir = work_dir + "/%s_rst/"%(start_pdb.split('/')[-1])
        if not os.path.isdir(rst_dir): os.mkdir(rst_dir)
        seq = Utils.get_seqs_from_pdb(start_pdb)
        pots = gen_pot_from_pred(seq, item)
        rosetta_rsts = gen_RosettaRst_from_pot(pots, rst_dir)

        for i in range(n_decoy):
            refined_pdb = work_dir + "/%s.R%d.pdb"%(start_pdb.split('/')[-1], i+1)
            print(start_pdb, refined_pdb)
            # start_pdb, refined_pdb, refined_energy = folding_by_FastRelax(start_pdb, rosetta_rsts, refined_pdb)
            process_results.append(process_pool.apply_async(folding_by_FastRelax, (start_pdb, rosetta_rsts, refined_pdb)))

    process_pool.close()
    process_pool.join()

    # collect results
    refined_results = {}
    for res in process_results:
        start_pdb, refined_pdb, refined_energy = res.get()[0], res.get()[1], res.get()[2]
        if start_pdb not in refined_results: refined_results[start_pdb] = {}
        refined_results[start_pdb][refined_pdb] = refined_energy
    
    le_pdbs = {} # lowest energy pdbs
    for start_pdb in refined_results:
        le_pdb = sorted(refined_results[start_pdb].items(), key=lambda kv: kv[1])[0][0]
        le_pdbs[start_pdb] = le_pdb

    return le_pdbs
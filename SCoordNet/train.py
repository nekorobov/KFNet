import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import tensorflow as tf
from model import run_training, run_testing, FLAGS
from tools.io import get_snapshot, get_num_trainable_params
from tensorflow.python import debug as tf_debug
from cnn_wrapper import helper, SCoordNet
from datetime import datetime
import yaml
import numpy as np
from eval import dist_error, read_lines, get_indexes, median_uncertainty

def set_stepvalue():
    if FLAGS.scene == 'scene01':
        FLAGS.stepvalue = 100000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'scene02':
        FLAGS.stepvalue = 100000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'scene03':
        FLAGS.stepvalue = 60000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'scene04':
        FLAGS.stepvalue = 100000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'scene05':
        FLAGS.stepvalue = 100000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'scene06':
        FLAGS.stepvalue = 100000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'scene07':
        FLAGS.stepvalue = 100000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'scene08':
        FLAGS.stepvalue = 100000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'scene09':
        FLAGS.stepvalue = 100000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    elif FLAGS.scene == 'scene10':
        FLAGS.stepvalue = 100000
        FLAGS.max_steps = FLAGS.stepvalue * 5
    else:
        print 'Invalid scene:', FLAGS.scene
        exit()


def load_intrinsics(path_to_cam_file, spec):
    with open(path_to_cam_file, 'r') as stream:
        data_loaded = yaml.safe_load(stream)["camera_intrinsics"]

    spec.focal_x = data_loaded["model"][0]
    spec.focal_y = data_loaded["model"][1]
    spec.u = data_loaded["model"][2]
    spec.v = data_loaded["model"][3]
    return spec


def solver(loss):
    """Solver."""
    # Get weight variable list.
    weights_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # Apply regularization to variables.
    reg_loss = tf.contrib.layers.apply_regularization(
        tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay), weights_list)
    with tf.device('/device:CPU:0'):
        # Get global step counter.
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
    # Decay the learning rate exponentially based on the number of steps.
    lr_op = tf.train.exponential_decay(FLAGS.base_lr,
                                       global_step=global_step,
                                       decay_steps=FLAGS.stepvalue,
                                       decay_rate=FLAGS.gamma,
                                       name='lr')
    # Get the optimizer. Moving statistics are added to optimizer.
    bn_list = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # For networks with batch normalization layers, it is necessary to
    # explicitly fetch its moving statistics and add them to the optimizer.
    with tf.control_dependencies(bn_list):
        opt = tf.train.AdamOptimizer(learning_rate=lr_op).minimize(
            loss + reg_loss, global_step=global_step)
    return opt, lr_op, reg_loss

def run_eval(sess, writer, scoordnet, spec, step, indexes, images, coords, uncertainty, gt_coords, mask, image_indexes):
    loop_num = len(indexes) / spec.batch_size
    coeffs = []
    dists = []
    median_uncertainties = []
    accuracies = []

    for i in range(loop_num):
        batch_images, batch_coords, batch_uncertainty, batch_gt_coords, batch_mask, batch_image_indexes = \
            sess.run([images, coords, uncertainty, gt_coords, mask, image_indexes])

        for b in range(spec.batch_size):
            id = i * spec.batch_size + b
            index = indexes[id]

            out_images = batch_images[b, :, :, :]
            out_uncertainty = batch_uncertainty[b, :, :, :]
            out_mask = batch_mask[b, :, :, :]
            out_coords = batch_coords[b, :, :, :]
            out_gt_coords = batch_gt_coords[b, :, :, :]
            out_weights = 1.0 / out_uncertainty
            out_image_index = batch_image_indexes[b]

            median_dist, euc_diff_coords, accuracy = dist_error(out_coords, out_gt_coords, out_mask)
            dists.append(median_dist)
            med_uncertainty = median_uncertainty(out_uncertainty, out_mask)
            median_uncertainties.append(med_uncertainty)
            accuracies.append(accuracy)

            print index, out_image_index, ', median dist:', median_dist, \
                ', median uncertainty:', med_uncertainty, \
                ', accuracy:', accuracy

    def write_summary(tag, value):
        summary = tf.Summary(value=[
                tf.Summary.Value(tag=tag, simple_value=value), 
                ])
        writer.add_summary(summary=summary, global_step=step)
       
    write_summary('median_coord_error(cm)', np.median(dists))
    write_summary('mean_coord_error(cm)', np.mean(dists))
    write_summary('stddev_coord_error(cm)', np.std(dists))
    write_summary('median_uncertainty', np.median(median_uncertainties))
    write_summary('mean_accuracy', np.mean(accuracies))


def train(image_list, label_list, transform_file, camera_file, image_list_eval, label_list_eval, out_dir, \
          snapshot=None, init_step=0, debug=False):

    print image_list
    print label_list
    if FLAGS.reset_step >=0:
        init_step = FLAGS.reset_step

    spec = helper.get_data_spec(model_class=SCoordNet)
    spec.scene = FLAGS.scene
    set_stepvalue()

    spec = load_intrinsics(camera_file, spec)

    print "--------------------------------"
    print "scene:", spec.scene
    print "batch size: ", spec.batch_size
    print "step value: ", FLAGS.stepvalue
    print "max steps: ", FLAGS.max_steps
    print "current step: ", init_step

    raw_input("Please check the meta info, press any key to continue...")

    scoordnet, loss, coord_loss, smooth_loss, accuracy, batch_indexes = run_training(image_list, label_list, transform_file)

    image_paths_eval = read_lines(image_list_eval)
    image_num_eval = len(image_paths_eval)

    indexes_eval = get_indexes(image_num_eval)
    
    _, images_eval, coords_eval, uncertainty_eval, gt_coords_eval, mask_eval, image_indexes_eval = \
        run_testing(indexes_eval, image_list_eval, label_list_eval, transform_file, spec, scoordnet=scoordnet)

    print '# trainable parameters: ', get_num_trainable_params()

    #mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/device:GPU:{}".format(int(d)) for d in FLAGS.devices])
    #with mirrored_strategy.scope():
    with tf.device('/device:GPU:%d' % int(FLAGS.devices[0])):
        optimizer, lr_op, reg_loss = solver(loss)
        init_op = tf.global_variables_initializer()

    # configuration
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)
    summary_op = tf.summary.merge_all()

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        if debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        summary_writer = tf.summary.FileWriter(FLAGS.model_folder + '/log', sess.graph)

        # Initialize variables.
        print('Running initializaztion operator.')
        sess.run(tf.global_variables_initializer())
        step = init_step

        # Start populating the queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        if snapshot:
            print('Pre-trained model restored from %s' % (snapshot))
            restore_variables = tf.global_variables()
            restorer = tf.train.Saver(restore_variables)
            restorer.restore(sess, snapshot)
            vars = {v.name: v for v in restore_variables}
            assign_step = vars['global_step:0'].assign(tf.constant(init_step))
            sess.run(assign_step)

        while step <= FLAGS.max_steps:
            start_time = time.time()
            summary, _, out_loss, out_coord_loss, out_smooth_loss, out_accuracy, out_indexes, lr = \
                sess.run([summary_op, optimizer, loss, coord_loss, smooth_loss, accuracy, batch_indexes, lr_op])
            duration = time.time() - start_time

            # Print info.
            if step % FLAGS.display == 0 or not FLAGS.is_training:
                summary_writer.add_summary(summary, step)
                format_str = '[%s] step %d/%d, %4d~%4d~%4d~%4d, loss = %.4f, coord_loss = %.4f, smooth_loss = %.4f, accuracy = %.4f, lr = %.6f (%.3f sec/step)'
                print(format_str % (datetime.now(), step, FLAGS.max_steps,
                                                 out_indexes[-4], out_indexes[-3], out_indexes[-2], out_indexes[-1],
                                                 out_loss, out_coord_loss, out_smooth_loss, out_accuracy,
                                                 lr, duration))

            # Save the model checkpoint periodically.
            if step % FLAGS.snapshot == 0 or step == FLAGS.max_steps:
                checkpoint_path = os.path.join(out_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

            if step % 1000 == 0:
                run_eval(sess, summary_writer, scoordnet, spec, step, indexes_eval, images_eval, coords_eval, uncertainty_eval, gt_coords_eval, mask_eval, image_indexes_eval)
            step += 1

        coord.request_stop()
        coord.join(threads)

def main(_):
    if FLAGS.finetune_folder != '':
        snapshot, step = get_snapshot(FLAGS.finetune_folder)
        step = 0
    else:
        snapshot, step = get_snapshot(FLAGS.model_folder)

    image_list = os.path.join(FLAGS.input_folder, 'image_list.txt')
    label_list = os.path.join(FLAGS.input_folder, 'label_list.txt')
    transform_file = os.path.join(FLAGS.input_folder, 'transform.txt')
    camera_file = os.path.join(FLAGS.input_folder, 'camera.yaml')

    image_list_eval = os.path.join(FLAGS.input_folder_eval, 'image_list.txt')
    label_list_eval = os.path.join(FLAGS.input_folder_eval, 'label_list.txt')

    train(image_list, label_list, transform_file, camera_file, image_list_eval, label_list_eval, FLAGS.model_folder,
          snapshot, step, FLAGS.debug)


if __name__ == '__main__':
    tf.app.run()

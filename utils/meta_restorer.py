import tensorflow as tf
from tensorflow.python.framework.meta_graph import import_scoped_meta_graph, copy_scoped_meta_graph, export_scoped_meta_graph

def better_meta_load(path, input_map={}, import_scope='', trainable=False, clear_devices=True, restore_vars=True):
    assert import_scope, 'Provide import scope! You will not regret it.'
    vars = import_scoped_meta_graph('%s.meta'%path, clear_devices=clear_devices,
                                        input_map=input_map, import_scope=import_scope)
    if not trainable:
        trainable_collection = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in tf.global_variables():
            if not var.name.startswith(import_scope+'/'):
                continue
            found = None
            for e in trainable_collection:
                if e.name == var.name:
                    found = e
                    break
            if found is not None:
                trainable_collection.remove(found)
    return better_restore(path, restore_scope=import_scope) if restore_vars else None

def better_restore(file_name, restore_scope=''):
    """ returns restore op
    """
    restore_scope = restore_scope if not restore_scope else restore_scope +'/'

    reader = tf.train.NewCheckpointReader('%s.ckpt' % file_name)
    strip_len = len(restore_scope)
    restore_ops = []
    for var in tf.global_variables():
        if var.name.startswith(restore_scope):
            #print 'restoring', var.name
            saved_var = reader.get_tensor(var.name[strip_len:-2])
            restore_ops.append(tf.assign(var, saved_var))
        else:
            pass
           # print 'did not restore', var.name
    return tf.group(*restore_ops)


def restore_in_scope(file_name, scope, prepend_scope_name_during_lookup=False):
    """ Restores as many variables in scope as possible. Anything found in checkpoint will be used :) """
    if file_name is None:
        return tf.no_op()
    to_restore = tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES, scope=scope if scope else None)
    print len(to_restore), 'to restore'
    reader = tf.train.NewCheckpointReader(('%s.ckpt' % file_name) if not file_name.endswith('.ckpt') else file_name)

    restore_ops = []
    for var in to_restore:
        name_to_lookup = var.name[:-2] if prepend_scope_name_during_lookup else var.name[(len(scope)+1) if scope else 0:-2]
        try:
            saved_var = reader.get_tensor(name_to_lookup)
        except:
            print 'Variable %s not found in the checkpoint!' % var.name
            continue
        restore_ops.append(tf.assign(var, saved_var))

    return tf.group(*restore_ops)




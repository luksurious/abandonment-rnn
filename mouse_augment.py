import numpy as np
from keras_preprocessing.sequence import pad_sequences


def augment_coord(x, y, only0=False, only1=False, cutoff=True, varycoord=True, varycutoff=False, varycount=3,
                  cutoff_list=None, cutoff_limit=2.5, cutoff_end=True, balance=True, offset_dupes=False,
                  var_strength=2):
    if cutoff_list is None:
        cutoff_list = [2, 3, 4]

    x_new, y_new = [], []
    for inp, lab in zip(x, y):

        if only0 and lab == 1:
            continue

        if only1 and lab == 0:
            continue

        # remove padding
        new_inp = [coords.tolist() for coords in inp if max(coords) > 0]

        if varycoord:
            for _ in range(varycount):
                # create new variations from each sample
                new_inp_rand = create_copy_with_varying_coords(new_inp, var_strength)

                x_new.append(new_inp_rand)
                y_new.append(lab)

        if cutoff:
            for cut_off_len in cutoff_list:
                if len(new_inp) > cutoff_limit * cut_off_len:
                    # create new data by cutting off some elements in the end
                    if cutoff_end:
                        new_inp_cut = new_inp[:-cut_off_len].copy()

                        x_new.append(new_inp_cut)
                        y_new.append(lab)

                    # cut off from beginning
                    new_inp_cut = new_inp[cut_off_len:].copy()

                    x_new.append(new_inp_cut)
                    y_new.append(lab)

        if varycutoff:
            for _ in range(varycount):
                # create new variations from each sample
                new_inp_rand = create_copy_with_varying_coords(new_inp, var_strength)

                for cut_off_len in cutoff_list:
                    if len(new_inp_rand) > cutoff_limit * cut_off_len:
                        # create new data by cutting off some elements in the end
                        if cutoff_end:
                            new_inp_cut = new_inp_rand[:-cut_off_len].copy()

                            x_new.append(new_inp_cut)
                            y_new.append(lab)

                        # cut off from beginning
                        new_inp_cut = new_inp_rand[cut_off_len:].copy()

                        x_new.append(new_inp_cut)
                        y_new.append(lab)

                x_new.append(new_inp_rand)
                y_new.append(lab)

        if offset_dupes:
            new_inp[:, 0] += np.random.randint(-20, 20)
            new_inp[:, 1] += np.random.randint(-20, 20)
            x_new.append(new_inp)
            y_new.append(lab)

    x_new = pad_sequences(x_new, maxlen=x[0].shape[0])
    x_new = np.array(x_new)
    y_new = np.array(y_new)

    if balance:
        class_1_count = np.count_nonzero(y)
        class_0_count = len(y) - class_1_count

        new_class_1_count = np.count_nonzero(y_new)
        new_class_0_count = len(y_new) - new_class_1_count

        total_1 = class_1_count + new_class_1_count
        total_0 = class_0_count + new_class_0_count

        class0_over = total_0 - total_1
        class1_over = total_1 - total_0

        x_new, y_new = remove_superfluous(class0_over, 0, x_new, y_new)
        x_new, y_new = remove_superfluous(class1_over, 1, x_new, y_new)

    x = np.concatenate([x, x_new])
    y = np.concatenate([y, y_new])

    return x, y


def remove_superfluous(count_over, class_lab, x_new, y_new):
    if count_over > 0:
        new_indices = np.flatnonzero(y_new == class_lab)
        new_remove = np.random.choice(new_indices, size=count_over)
        y_new = np.delete(y_new, new_remove)
        x_new = np.delete(x_new, new_remove, axis=0)

    return x_new, y_new


def create_copy_with_varying_coords(inp, var_strength):
    new_inp_rand = []
    for coord in inp:
        new_coord = coord.copy()
        new_coord[0] = coord[0] + np.random.randint(-var_strength, var_strength)
        new_coord[1] = coord[1] + np.random.randint(-var_strength, var_strength)
        new_inp_rand.append(new_coord)

    return new_inp_rand

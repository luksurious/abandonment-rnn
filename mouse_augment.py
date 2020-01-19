import numpy as np
from keras_preprocessing.sequence import pad_sequences


def augment_coord_multi(X, y1, y2, y3, only0=False, only1=False, cutoff=True, varycoord=True):
    X_new, y1_new, y2_new, y3_new = [], [], [], []
    var_strength = 2
    for inp, lab1, lab2, lab3 in zip(X, y1, y2, y3):

        if only0 and lab1 == 1 and lab2 == 1 and lab3 == 1:
            continue

        if only1 and lab1 == 0 and lab2 == 0 and lab3 == 0:
            continue

        if varycoord:
            for _ in range(3):
                # create 3 new variations from each sample
                new_inp_rand = create_copy_with_varying_coords(inp, var_strength)

                X_new.append(new_inp_rand)
                y1_new.append(lab1)
                y2_new.append(lab2)
                y3_new.append(lab3)

        if cutoff:
            for cut_off in [2, 5]:
                if len(inp) > 5 * cut_off:
                    # create new data by cutting off some elements in the end
                    new_inp_cut = inp[:-cut_off].copy()

                    X_new.append(new_inp_cut)
                    y1_new.append(lab1)
                    y2_new.append(lab2)
                    y3_new.append(lab3)

                    # cut off from beginning
                    new_inp_cut = inp[cut_off:].copy()

                    X_new.append(new_inp_cut)
                    y1_new.append(lab1)
                    y2_new.append(lab2)
                    y3_new.append(lab3)

    X = np.concatenate([X, np.array(X_new)])
    y1 = np.concatenate([y1, np.array(y1_new)])
    y2 = np.concatenate([y2, np.array(y2_new)])
    y3 = np.concatenate([y3, np.array(y3_new)])

    return X, y1, y2, y3


def augment_coord(x, y, only0=False, only1=False, cutoff=True, varycoord=True, varycutoff=False, varycount=3,
                  cutoff_list=None, cutoff_limit=2.5, cutoff_end=True):
    if cutoff_list is None:
        cutoff_list = [2, 3, 4]

    x_new, y_new = [], []
    var_strength = 2
    for inp, lab in zip(x, y):

        if only0 and lab == 1:
            continue

        if only1 and lab == 0:
            continue

        new_inp = [coords.tolist() for coords in inp if max(coords) > 0]

        if varycoord:
            for _ in range(varycount):
                # create 3 new variations from each sample
                new_inp_rand = create_copy_with_varying_coords(new_inp, var_strength)

                x_new.append(new_inp_rand)
                y_new.append(lab)

        if cutoff:
            for cut_off in cutoff_list:
                if len(new_inp) > cutoff_limit * cut_off:
                    # create new data by cutting off some elements in the end
                    if cutoff_end:
                        new_inp_cut = new_inp[:-cut_off].copy()

                        x_new.append(new_inp_cut)
                        y_new.append(lab)

                    # cut off from beginning
                    new_inp_cut = new_inp[cut_off:].copy()

                    x_new.append(new_inp_cut)
                    y_new.append(lab)

        if varycutoff:
            for _ in range(varycount):
                # create 3 new variations from each sample
                new_inp_rand = create_copy_with_varying_coords(new_inp, var_strength)

                for cut_off in cutoff_list:
                    if len(new_inp_rand) > cutoff_limit * cut_off:
                        # create new data by cutting off some elements in the end
                        if cutoff_end:
                            new_inp_cut = new_inp_rand[:-cut_off].copy()

                            x_new.append(new_inp_cut)
                            y_new.append(lab)

                        # cut off from beginning
                        new_inp_cut = new_inp_rand[cut_off:].copy()

                        x_new.append(new_inp_cut)
                        y_new.append(lab)

                x_new.append(new_inp_rand)
                y_new.append(lab)

    x_new = pad_sequences(x_new, maxlen=x[0].shape[0])

    x = np.concatenate([x, np.array(x_new)])
    y = np.concatenate([y, np.array(y_new)])

    return x, y


def create_copy_with_varying_coords(inp, var_strength):
    new_inp_rand = []
    for coord in inp:
        # if np.random.random() < 0.5:
        # create new data by moving the positions a bit
        new_coord = coord.copy()
        new_coord[0] = coord[0] + np.random.randint(-var_strength, var_strength)
        new_coord[1] = coord[1] + np.random.randint(-var_strength, var_strength)
        new_inp_rand.append(new_coord)
        # else:
        #    new_inp_rand.append(coord)

    return new_inp_rand

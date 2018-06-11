import matplotlib.pyplot as plt
import time

class LogHelper():
    def __init__(self):
        localtime = time.localtime(time.time())
        t = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.file_name = "log/" + str(t) + "_"

    # def write_log(self, epoch, epochs, iteration, log, tag):
    #     with open(self.file_name + "log.txt", 'a') as file:
    #         file.write("Epoch {}/{}".format(epoch, epochs))
    #         file.write("Iteration: {}".format(iteration))
    #         file.write(" " + tag + ":{:.4f}".format(log))
    #         file.write("\n")
    #     self.print_log(epoch, epochs, iteration, log, tag)
    #
    # def print_log(self, epoch, epochs, iteration, log, tag):
    #     print("Epoch {}/{}".format(epoch, epochs),
    #           "Iteration: {}".format(iteration),
    #           tag + ": {:.4f}".format(log))

    def save_plt(self, x1, x2, file_name, x1_label, x2_label, y_label):
        line_1, = plt.plot(x1, 'b', label=x1_label)
        line_2, = plt.plot(x2, 'g', label=x2_label)
        # line_down, = plt.plot(time, down, 'r', label='Line down')

        # plt.title(file_name)
        plt.xlabel("epoch")
        plt.ylabel(y_label)
        plt.legend(handles=[line_1, line_2])

        plt.savefig(self.file_name + file_name + '_' + y_label)
        plt.clf()
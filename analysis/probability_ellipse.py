import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class ConfidenceEllipse:
    def __init__(self, data, p=0.95):
        self.data = data
        self.p = p

        # 平均値
        self.means = np.mean(data, axis=0)
        # 分散行列
        self.cov = np.cov(data[:,0], data[:,1])
        

        # 固有値，固有値ベクトルの求解
        lambdas, vecs = np.linalg.eigh(self.cov)
        # 第1主成分から順番通りになるよう，固有値が大きい順にソート
        order = lambdas.argsort()[::-1]
        lambdas, vecs = lambdas[order], vecs[:,order]

        c = np.sqrt(chi2.ppf(self.p, 2))
        # 確率楕円の幅，高さ
        self.w, self.h = 2 * c * np.sqrt(lambdas)
        # 確率楕円の傾き
        self.theta = np.degrees(np.arctan(
            ((lambdas[0] - lambdas[1])/self.cov[0,1])))
        
    def get_params(self):
        return self.means, self.w, self.h, self.theta

    def get_point(self):
        p_orig = [[0.5 * self.w * np.cos(0.25*np.pi*float(i)), 0.5 * self.h * np.sin(0.25*np.pi*float(i))] for i in range(8)]

        theta_rad = np.deg2rad(self.theta)
        rot_mat = np.array([[np.cos(theta_rad), - np.sin(theta_rad)],
                            [np.sin(theta_rad),   np.cos(theta_rad)]])
        point = [self.means + rot_mat.dot(p) for p in p_orig]

        return np.array(point)

    def get_patch(self, line_color="black", face_color="none", alpha=0):
        el = Ellipse(xy=self.means,
                     width=self.w, height=self.h,
                     angle=self.theta, color=line_color, alpha=alpha)
                     #angle=0.0, color=line_color, alpha=alpha)
        el.set_facecolor(face_color)
        return el

def main():
    #data = gen_data()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(data[:,0], data[:,1], color="b", marker=".", s=3)

    el = ConfidenceEllipse(data, p=0.95)
    means, w, h, theta = el.get_params()
    p1 = np.array([0.5*w, 0.0])
    p2 = np.array([-0.5*w, 0.0])
    p3 = np.array([0.0, 0.5*h])
    p4 = np.array([0.0, -0.5*h])
    angle = np.radians(theta)
    rotmat = np.array([[np.cos(angle), - np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
    p1_ = rotmat.dot(p1) + means
    p2_ = rotmat.dot(p2) + means
    p3_ = rotmat.dot(p3) + means
    p4_ = rotmat.dot(p4) + means
    print(means, w, h, theta)
    print(p1_, p2_, p3_, p4_)
    ax.add_artist(el.get_patch(face_color="blue", alpha=0.5))
    ax.scatter(means[0],means[1], color="red", marker="o", s=10)
    ax.scatter(p1_[0], p1_[1], color="gold", marker="o", s=10)
    ax.scatter(p2_[0], p2_[1], color="gold", marker="o", s=10)
    ax.scatter(p3_[0], p3_[1], color="gold", marker="o", s=10)
    ax.scatter(p4_[0], p4_[1], color="gold", marker="o", s=10)
    ax.scatter(p1[0]+means[0], p1[1]+means[1], color="g", marker="o", s=10)
    ax.scatter(p2[0]+means[0], p2[1]+means[1], color="g", marker="o", s=10)
    ax.scatter(p3[0]+means[0], p3[1]+means[1], color="g", marker="o", s=10)
    ax.scatter(p4[0]+means[0], p4[1]+means[1], color="g", marker="o", s=10)
    ax.add_artist(Ellipse(xy=means,
                          width=w, height=h,
                          angle=0.0, color='r', alpha=0.5))
    #plt.savefig("img.png")
    ax.set_aspect('equal')
    plt.show()

if __name__ == "__main__":
    #main()
    #data = gen_data()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(data[:,0], data[:,1], color="b", marker=".", s=3)

    el = ConfidenceEllipse(data, p=0.95)
    means, w, h, theta = el.get_params()
    p1 = np.array([0.5*w, 0.0])
    p2 = np.array([-0.5*w, 0.0])
    p3 = np.array([0.0, 0.5*h])
    p4 = np.array([0.0, -0.5*h])
    angle = np.radians(theta)
    rotmat = np.array([[np.cos(angle), - np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]])
    p1_ = rotmat.dot(p1) + means
    p2_ = rotmat.dot(p2) + means
    p3_ = rotmat.dot(p3) + means
    p4_ = rotmat.dot(p4) + means
    print(means, w, h, theta)
    print(p1_, p2_, p3_, p4_)
    ax.add_artist(el.get_patch(face_color="blue", alpha=0.5))
    ax.scatter(means[0],means[1], color="red", marker="o", s=10)
    ax.scatter(p1_[0], p1_[1], color="gold", marker="o", s=10)
    ax.scatter(p2_[0], p2_[1], color="gold", marker="o", s=10)
    ax.scatter(p3_[0], p3_[1], color="gold", marker="o", s=10)
    ax.scatter(p4_[0], p4_[1], color="gold", marker="o", s=10)
    ax.scatter(p1[0]+means[0], p1[1]+means[1], color="g", marker="o", s=10)
    ax.scatter(p2[0]+means[0], p2[1]+means[1], color="g", marker="o", s=10)
    ax.scatter(p3[0]+means[0], p3[1]+means[1], color="g", marker="o", s=10)
    ax.scatter(p4[0]+means[0], p4[1]+means[1], color="g", marker="o", s=10)
    ax.add_artist(Ellipse(xy=means,
                          width=w, height=h,
                          angle=0.0, color='r', alpha=0.5))
    #plt.savefig("img.png")
    ax.set_aspect('equal')
    plt.show()


sample_gsd = 1.05 #(mm/pixel) dji mini 2 4K
sample_dis = 3.0 # (m)

def cal_gsd(work_dis):
    img_gsd = sample_gsd * work_dis / sample_dis
    print("GSD: {:.3f}".format(img_gsd), "mm/pixel")

    return img_gsd


#cal_gsd(5.0)




plt.figure
plt.imshow(r)

r = img[:,:,0]

g = img[:,:,1]

b = img[:,:,2]

nlf = 8-n%8
lt = x[nl-1,:][np.newaxis,:]
ct = x[:,nl-1][:,np.newaxis]

newl = lt.reapeat(nlf,axis=0)
newc = ct.repeat(nlf,axis=1)

xp = np.vstack((x,newl))


## 6 cv2.resize

'''
75%
YQ 158
    -1
    0


Yb_DPCM 1
    -1
    0

Yb_iDPCM 158
    -1
    0




MAX DIFF 43

'''
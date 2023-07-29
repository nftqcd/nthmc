if __name__=='__main__':
    import sys
    import numpy
    import tensorflow as tf
    from nthmc.lib import gauge, group, lattice, field, fieldio

    fn = sys.argv[1]
    g,lat = fieldio.readLattice(fn)
    ndim = len(lat)
    tf.print(g.shape)
    tf.print(lat)

    # reverse read
    g = numpy.transpose(g, axes=list(range(1, ndim+1))+[0]+list(range(ndim+1, ndim+3)))
    # now g is in the wrong storage order

    # the wrong transpose in previous writeLattice would leave the wrong shape as
    g = numpy.reshape(g, (lat[3], ndim, lat[0], lat[1], lat[2], 3, 3))

    # reverse the wrong transpose in previous writeLattice (above again)
    g = numpy.transpose(g, axes=list(range(1, ndim+1))+[0]+list(range(ndim+1, ndim+3)))
    # now g is in the correct in-memory order

    # check plaquette
    gg = gauge.Gauge([gauge.Transporter(lattice.Lattice(tf.constant(g[i]),nd=ndim,batch_dim=-1), field.Path(i+1)) for i in range(ndim)])
    for pl in gauge.plaquette(gg):
        print(pl.numpy())
    tf.print('plaq:',gauge.plaq(gg))

    outfn = fn+'.fixed.lime'
    print(f'writing out: {outfn}')
    fieldio.writeLattice(g, outfn)

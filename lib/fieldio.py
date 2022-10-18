import numpy
import datetime,os,re,struct,sys,time,zlib

limeMagic = 0x456789ab

def limeHeader(mbeg, mend, size, type):
    m = 0
    if mbeg:
        m = 1<<15
    elif mend:
        m = 1<<14
    return struct.pack('>ihHq128s', limeMagic, 1, m, size, type)

def limeItemWrite(f, mbeg, mend, bytes, type):
    size = len(bytes)
    header = limeHeader(mbeg, mend, size, type)
    pad = 7-(size+7)%8
    f.write(header)
    f.write(bytes)
    f.write(b'\0' * pad)

def scidacChecksum(latdata, vol, sitesize):
    suma = 0
    sumb = 0
    for s in range(vol):
        c = zlib.crc32(latdata[s*sitesize:(s+1)*sitesize]) & 0xffffffff
        s29 = s%29
        s31 = s%31
        suma ^= (c<<s29 | c>>(32-s29)) & 0xffffffff
        sumb ^= (c<<s31 | c>>(32-s31)) & 0xffffffff
    return suma,sumb

def xmlFind(s,r):
    rs = '<'+r+r'>([^<]*)</'+r+'>'
    m = re.findall(rs.encode('utf-8'),s)
    if len(m)!=1:
        raise ValueError(f'finding record {r} in unsupported xml: {s}')
    return m[0]

def printLimeHeaders(file):
    f = open(file,'rb')
    while True:
        header = f.read(144)
        if not header:
            break
        magi, vers, mbeg_end_res, size, type = struct.unpack('>ihHq128s',header)
        if magi!=limeMagic:
            raise IOError(f'lime magic number does not match, got {magi}')
        print(f'* HEADER vers: {vers}')
        mbeg = 1 & mbeg_end_res>>15
        mend = 1 & mbeg_end_res>>14
        mres = ~(3<<14) & mbeg_end_res
        print(f'mbeg mend res: {mbeg} , {mend} , {mres}')
        type = type[:type.find(b'\0')]
        print(f'size: {size}')
        print(f'type: {type}')
        loca = f.tell()
        print(f'loca: {loca}')
        if type==b'scidac-binary-data' or type==b'ildg-binary-data':
            next = (size+7)//8*8
        else:
            data = f.read(size)
            data = data[:data.find(b'\0')]
            print(f'* DATA: {data}')
            next = 7-(size+7)%8
        # print(f'next: {next}')
        f.seek(next,os.SEEK_CUR)
    f.close()

def readLattice(file, verbose=True):
    """
    Only supports SciDAC and ILDG formats in Lime.
    """
    time0 = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    def logput(*args,**kwargs):
        if verbose:
            print(*args,**kwargs)
    f = open(file,'rb')
    latsize = -1
    lattypesize = -1
    latdatacount = -1
    latdims = []
    latnc = -1
    latns = -1
    latprec = -1
    lattype = b'Unknown'
    while True:
        header = f.read(144)
        if not header:
            break
        magi, vers, mbeg_end_res, size, type = struct.unpack('>ihHq128s',header)
        if magi!=limeMagic:
            raise IOError(f'lime magic number does not match, got {magi}')
        # logput(f'vers: {vers}')
        # mbeg = 1 & mbeg_end_res>>15
        # mend = 1 & mbeg_end_res>>14
        # mres = ~(3<<14) & mbeg_end_res
        # logput(f'mbeg mend res: {mbeg} , {mend} , {mres}')
        type = type[:type.find(b'\0')]
        # logput(f'size: {size}')
        # logput(f'type: {type}')
        loca = f.tell()
        # logput(f'loca: {loca}')
        if type==b'scidac-binary-data' or type==b'ildg-binary-data':
            lattype = type
            latsize = size
            latdata = f.read(size)
            next = 7-(size+7)%8
        else:
            data = f.read(size)
            data = data[:data.find(b'\0')]
            if type==b'scidac-private-file-xml':
                # logput('version: ',xmlFind(data,'version'))
                spacetime = int(xmlFind(data,'spacetime'))
                latdims = [int(x) for x in xmlFind(data,'dims').strip().split()]
                if len(latdims)!=spacetime:
                    raise ValueError(f'got spacetime {spacetime} but dims {latdims}')
            elif type==b'scidac-file-xml':
                logput(f'file metadata: {data}')
            elif type==b'scidac-private-record-xml':
                # logput('version: ',xmlFind(data,'version'))
                logput('date: ',xmlFind(data,'date'))
                logput('recordtype: ',xmlFind(data,'recordtype'))
                logput('datatype: ',xmlFind(data,'datatype'))
                precision = xmlFind(data,'precision').lower()
                if precision==b'f':
                    latprec = 8    # complex float
                elif precision==b'd':
                    latprec = 16   # complex double
                else:
                    raise ValueError(f'unknown precision {precision}')
                latnc = int(xmlFind(data,'colors'))
                try:
                    latns = int(xmlFind(data,'spins'))
                except ValueError as err:
                    logput(f'Ignore exceptions in finding "spins" in xml: {err}')
                    latns = 1
                lattypesize = int(xmlFind(data,'typesize'))
                latdatacount = int(xmlFind(data,'datacount'))
            elif type==b'scidac-record-xml':
                logput(f'record metadata: {data}')
            elif type==b'scidac-checksum':
                # logput('version: ',xmlFind(data,'version'))
                latsuma = int(xmlFind(data,'suma'),16)
                latsumb = int(xmlFind(data,'sumb'),16)
                pass
            elif type==b'ildg-format':
                field = xmlFind(data,'field')
                if field!=b'su3gauge':
                    raise ValueError(f'unsupported ildg field type {field}')
                precision = int(xmlFind(data,'precision'))
                if precision==32:
                    latprec = 8    # complex float
                elif precision==64:
                    latprec = 16   # complex double
                else:
                    raise ValueError(f'unknown precision {precision}')
                lx = int(xmlFind(data,'lx'))
                ly = int(xmlFind(data,'ly'))
                lz = int(xmlFind(data,'lz'))
                lt = int(xmlFind(data,'lt'))
                latdims = [lx,ly,lz,lt]
                latnc = 3
                latns = 1
                lattypesize = latnc*latnc*latprec
                latdatacount = 4
            else:
                logput(f'unused type: {type}  data: {data}')
            next = 7-(size+7)%8
        # logput(f'next: {next}')
        f.seek(next,os.SEEK_CUR)
    f.close()
    time1 = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    if latsize<=0 or lattypesize<=0 or latdatacount<=0 or len(latdims)==0 or latnc<0 or latns<0 or latprec<0 or lattype==b'Unknown':
        raise ValueError(f'unsupported file: {file}')
    vol = 1
    for x in latdims:
        vol *= x
    ndim = len(latdims)
    if latsize==vol*latdatacount*lattypesize:
        if lattype==b'scidac-binary-data':
            if latsuma is None or latsumb is None:
                raise ValueError(f'No SciDAC checksum.')
            else:
                suma,sumb = scidacChecksum(latdata, vol, latdatacount*lattypesize)
                # logput(f'computed sum {suma:x} {sumb:x}')
                # logput(f'expected sum {latsuma:x} {latsumb:x}')
                if suma!=latsuma or sumb!=latsumb:
                    raise IOError(f'Checksum error: expected {latsuma:x} {latsumb:x}, computed {suma:x} {sumb:x}')
    else:
        raise ValueError(f'incorrect lattice size, expect {vol*latdatacount*lattypesize}, but got {latsize}')
    if latnc>0 and latns==1 and (latprec==8 or latprec==16) and lattypesize==latprec*latnc*latnc:
        lat = numpy.frombuffer(latdata,dtype='>c'+str(latprec),count=vol*latdatacount*latnc*latnc)
        if lattype==b'scidac-binary-data' or lattype==b'ildg-binary-data':
            lat = numpy.reshape(lat,latdims[::-1]+[latdatacount,latnc,latnc])
            # lat = numpy.transpose(lat, axes=list(range(ndim,-1,-1))+list(range(ndim+1,ndim+3)))
            # move latdatacount to the front, keep TZYX order
            lat = numpy.transpose(lat, axes=[ndim]+list(range(ndim))+list(range(ndim+1,ndim+3)))
        else:
            raise ValueError(f'unknown lattice format: {lattype}')
    else:
        raise ValueError(f'unsupported contents in file: {file}')
    time2 = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
    logput(f'read time: io {(time1-time0)*1e-6} ms, proc {(time2-time1)*1e-6} ms')
    return lat,latdims

def writeLattice(gauge, file):
    """
    Only writes in SciDAC format in Lime.
    """
    latdatacount = gauge.shape[0]
    latdims = gauge.shape[1:latdatacount+1]
    latnc = gauge.shape[latdatacount+1]
    if latnc!=gauge.shape[latdatacount+2]:
        raise ValueError(f'unsupported data shape: {gauge.shape}')
    if gauge.dtype.kind!='c':
        raise ValueError(f'unsupported data type: {gauge.dtype}')
    latprec = gauge.dtype.itemsize
    if latprec==8:
        precision = 'F'
    elif latprec==16:
        precision = 'D'
    else:
        raise ValueError(f'unknown dtype item size {gauge.dtype}')
    lattypesize = latprec*latnc*latnc
    vol = 1
    dims = ''
    for d in latdims:
        vol *= d
        dims += str(d) + ' '
    latsize = vol*latdatacount*lattypesize
    lattype = b'scidac-binary-data'

    ndim = len(latdims)
    # gauge = numpy.transpose(gauge, axes=list(range(ndim,-1,-1))+list(range(ndim+1,ndim+3)))
    # move ndim to the back, keep TZYX order
    gauge = numpy.transpose(gauge, axes=[ndim]+list(range(ndim))+list(range(ndim+1,ndim+3)))
    binary = numpy.ascontiguousarray(gauge, dtype=f'>c{latprec}').tobytes()
    suma,sumb = scidacChecksum(binary, vol, latdatacount*lattypesize)

    scidac_private_file = f'<?xml version="1.0" encoding="UTF-8"?><scidacFile><version>1.1</version><spacetime>{latdatacount}</spacetime><dims>{dims}</dims><volfmt>0</volfmt></scidacFile>'.encode()
    scidac_private_record = f'<?xml version="1.0" encoding="UTF-8"?><scidacRecord><version>1.1</version><date>{datetime.datetime.now(datetime.timezone.utc).ctime()} UTC</date><recordtype>0</recordtype><datatype>numpy.ndarray</datatype><precision>{precision}</precision><colors>{latnc}</colors><spins>1</spins><typesize>{lattypesize}</typesize><datacount>{latdatacount}</datacount></scidacRecord>'.encode()
    scidac_checksum = f'<?xml version="1.0" encoding="UTF-8"?><scidacChecksum><version>1.0</version><suma>{suma:x}</suma><sumb>{sumb:x}</sumb></scidacChecksum>'.encode()

    f = open(file,'wb')

    limeItemWrite(f, True, False, scidac_private_file, b'scidac-private-file-xml')
    limeItemWrite(f, False, True, b'<?xml version="1.0"?><note>generated by NTHMC</note>', b'scidac-file-xml')
    limeItemWrite(f, True, False, scidac_private_record, b'scidac-private-record-xml')
    limeItemWrite(f, False, False, b'<?xml version="1.0"?><note>gauge configuration</note>', b'scidac-record-xml')
    limeItemWrite(f, False, False, binary, b'scidac-binary-data')
    limeItemWrite(f, False, True, scidac_checksum, b'scidac-checksum')

    f.close()

if __name__=='__main__':
    def test_read(fn):
        printLimeHeaders(fn)
        g,lat = readLattice(fn)
        print(f'lattice: {lat}')
        return g,lat

    gconf,lat = test_read(sys.argv[1])

    import tensorflow as tf
    import group
    sys.path.append("../su3_4d")
    import gauge

    def check(g,lat):
        g = tf.expand_dims(g, axis=0)
        if g.dtype==tf.complex64:
            g = tf.cast(g, tf.complex128)
        for i in range(g.shape[1]):
            print(f'[0,0,0,0],{i},[0,0] {g[0,i,0,0,0,0,0,0].numpy()}')
            print(f'[0,0,0,1],{i},[0,0] {g[0,i,0,0,0,1,0,0].numpy()}')
            print(f'[0,0,1,0],{i},[0,0] {g[0,i,0,0,1,0,0,0].numpy()}')
            print(f'[0,1,0,0],{i},[0,0] {g[0,i,0,1,0,0,0,0].numpy()}')
            print(f'[1,0,0,0],{i},[0,0] {g[0,i,1,0,0,0,0,0].numpy()}')
        a,m = group.checkSU(g)
        print(f'checkSU avg: {a[0].numpy()} max: {m[0].numpy()}')
        act = gauge.SU3d4(tf.random.Generator.from_seed(1234567), nbatch=1,
            beta=0.7796, beta0=0.7796, c1=gauge.C1DBW2, size=lat)
        ps,_ = act.plaqFieldsWoTrans(g)
        for pl in ps:
            print(f'{pl.shape} first element {pl[0,0,0,0,0].numpy()}')
        for pl in act.plaquetteList(g):
            print(pl[0].numpy())
        print(f'action {act(g)}')
        f,_,_ = act.derivAction(g)
        for i in range(4):
            print(f'force norm2 dim {i} : {group.norm2(f[0,i],axis=range(6))}')
        print('projectSU')
        g = group.projectSU(g)
        a,m = group.checkSU(g)
        print(f'checkSU avg: {a[0].numpy()} max: {m[0].numpy()}')
        print(g.shape)
        for pl in act.plaquetteList(g):
            print(pl[0].numpy())
        return g[0].numpy()

    gconf = check(gconf,lat)

    outfn = sys.argv[1]+'.test'
    print(f'writing out: {outfn}')
    writeLattice(gconf, outfn)

    gconf,lat = test_read(outfn)
    check(gconf,lat)

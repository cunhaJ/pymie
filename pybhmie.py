#This is a simple function to calculate Mie scattering efficiencies 
#based on 1983 Bohren&Huffman Absorption and Scattering of Light by Small Particles

#requires scipy 
from scipy.special import jv
from scipy.special import yv


def pybhmie(x, m):
    '''Mie scattering by BHMIE
    Inputs 
    x: size parameters xmie = k*radius/medium = 
        = 2*np.pi*radius/(lamda/medium) with medium=n1/n2, ref index ratio
    m: complex index of refraction 
    
    Outputs
    Qext: Extinction efficiency (ext cross-section/ pi*radius**2)
    Qsca: Scattering efficiency (sca cross-section/ pi*radius**2)
    Qabs: Absorption efficiency (abs cross-section/ pi*radius**2)
    '''
    
    mx = m*x;
    
    #famous wiscombe series expansion stop criteria
    #each x has a different stopping value 
    nmax = numpy.array(x + 4.*x**(1.0/3.0) + 2., dtype=int)

    #create j vector, indexing the expansion members containing nmax
    j = np.arange(1,nmax+1)

    
    #Ricatti-Bessel functions with jv,yv 1st and 2nd Bessel functions
    psi = np.sqrt(pi*x/2)*jv(j+0.5,x)
    xi = np.sqrt(pi*x/2)*(jv(j+0.5,x) + 1j*yv(j+0.5,x))
    
    #psi(mx)
    psim = np.sqrt(pi*mx/2)*jv(j+0.5,mx)
    
    #calculate psi function series
    psi0 = np.array(psi[0:nmax-1])
    psi0 = np.insert(psi0,0,np.sin(x)) #initialize psi as in Appendix A of BH
    
    #print(psi)
    #print(phi0)
    
    #calculate psi(mx) function series
    psi0m = np.array(psim[0:nmax-1])
    psi0m = np.insert(psi0m,0,np.sin(mx))
    
    #calculate xi function series
    xi0 = np.array(xi[0:nmax-1])
    xi0 = np.insert(xi0,0,np.sin(x) - 1.0j*np.cos(x)) #initialize xi as in Appendix A of BH

    #derivatives calculated by downward recurrence
    dpsi = psi0 - j/x*psi
    dpsim = psi0m - j/mx*psim
    dxi = xi0 - j/x*xi

    #expansion coefficients
    aj = (m*psim*dpsi-psi*dpsim)/(m*psim*dxi-xi*dpsim)
    bj = (psim*dpsi-m*psi*dpsim)/(psim*dxi-m*xi*dpsim)

    #efficiencies calculated by sum of coeffs
    Qsca = sum( (2*j+1)*(abs(aj)*abs(aj) + abs(bj)*abs(bj)) )
    Qext = sum( (2*j+1)*np.real(aj + bj) )    

    Qext = Qext*2/(x*x)
    Qsca = Qsca*2/(x*x)
    Qabs = Qext - Qsca
    
    return Qext, Qsca, Qabs
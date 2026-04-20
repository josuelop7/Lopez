! ============================================================
! Module containing the kernels (device subroutines)
! ============================================================
module suma_kernels
    use cudafor
    implicit none
contains

    ! Kernel for vector addition (1D)
    attributes(global) subroutine suma_vector_kernel(a, b, c)
        real, device, intent(in)  :: a(:), b(:)
        real, device, intent(out) :: c(:)
        integer :: i, n
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        n = size(a)
        if (i <= n) then
            c(i) = a(i) + b(i)
        end if
    end subroutine suma_vector_kernel

    ! Kernel for matrix addition (2D)
    attributes(global) subroutine suma_matriz_kernel(A, B, C)
        real, device, intent(in)  :: A(:,:), B(:,:)
        real, device, intent(out) :: C(:,:)
        integer :: i, j
        i = (blockIdx%x - 1) * blockDim%x + threadIdx%x
        j = (blockIdx%y - 1) * blockDim%y + threadIdx%y
        if (i <= size(A,1) .and. j <= size(A,2)) then
            C(i,j) = A(i,j) + B(i,j)
        end if
    end subroutine suma_matriz_kernel

end module suma_kernels

! ============================================================
! Main program: first_task
! ============================================================
program first_task
    use cudafor
    use suma_kernels
    implicit none

    ! Parameters
    integer, parameter :: n_vec = 1024 * 64      ! Vector size
    integer, parameter :: n_mat = 128             ! Matrix dimension (square)
    integer, parameter :: tPB_vec = 256            ! Threads per block for vectors
    integer, parameter :: tPBx_mat = 16, tPBy_mat = 16   ! Threads per block for matrix (32x8 = 256)
    real, allocatable :: v1(:), v2(:), v_gpu(:), v_cpu(:)
    real, allocatable :: A(:,:), B(:,:), C_gpu(:,:), C_cpu(:,:)
    type(dim3) :: grid_vec, tBlock_vec
    type(dim3) :: grid_mat, tBlock_mat
    integer :: i, j
    integer :: ierr

    ! ============================================================
    ! 1. VECTOR ADDITION
    ! ============================================================
    allocate(v1(n_vec), v2(n_vec), v_gpu(n_vec), v_cpu(n_vec), stat=ierr)
    if (ierr /= 0) stop 'Vector alloc failed'

    ! Initialization (arbitrary values, as in class)
    v1 = [(real(i), i=1, n_vec)]
    v2 = [(real(n_vec - i + 1), i=1, n_vec)]

    ! Launch configuration for vectors
    tBlock_vec = dim3(tPB_vec, 1, 1)
    grid_vec = dim3(ceiling(real(n_vec) / tPB_vec), 1, 1)

    ! Copy data to GPU
    call suma_vector_gpu(v1, v2, v_gpu, grid_vec, tBlock_vec)

    ! Compute on CPU for verification
    call suma_vector_cpu(v1, v2, v_cpu)

    ! Compare results
    if (all(abs(v_gpu - v_cpu) <= 1.0e-5)) then
        print *, "VECTOR: GPU and CPU results match."
    else
        print *, "VECTOR: ERROR - Results do not match."
    end if

    deallocate(v1, v2, v_gpu, v_cpu)

    ! ============================================================
    ! 2. MATRIX ADDITION
    ! ============================================================
    allocate(A(n_mat, n_mat), B(n_mat, n_mat), C_gpu(n_mat, n_mat), C_cpu(n_mat, n_mat), stat=ierr)
    if (ierr /= 0) stop 'Matrix alloc failed'

    ! Initialization
    do i = 1, n_mat
        do j = 1, n_mat
            A(i,j) = real(i + j)
            B(i,j) = real(i * j)
        end do
    end do

    ! Launch configuration for matrices (2D)
    tBlock_mat = dim3(tPBx_mat, tPBy_mat, 1)
    grid_mat = dim3(ceiling(real(n_mat) / tBlock_mat%x), &
                    ceiling(real(n_mat) / tBlock_mat%y), 1)

    ! GPU addition
    call suma_matriz_gpu(A, B, C_gpu, grid_mat, tBlock_mat)

    ! CPU addition
    call suma_matriz_cpu(A, B, C_cpu)

    ! Compare
    if (all(abs(C_gpu - C_cpu) <= 1.0e-5)) then
        print *, "MATRIX: GPU and CPU results match."
    else
        print *, "MATRIX: ERROR - Results do not match."
    end if

    deallocate(A, B, C_gpu, C_cpu)

contains

    ! ------------------------------------------------------------
    ! Helper subroutine that manages device memory and launches kernel for vectors
    ! ------------------------------------------------------------
    subroutine suma_vector_gpu(a, b, c, grid, tBlock)
        real, intent(in)  :: a(:), b(:)
        real, intent(out) :: c(:)
        type(dim3), intent(in) :: grid, tBlock
        real, device, allocatable :: a_d(:), b_d(:), c_d(:)
        integer :: n
        n = size(a)
        allocate(a_d(n), b_d(n), c_d(n))
        a_d = a
        b_d = b
        call suma_vector_kernel<<<grid, tBlock>>>(a_d, b_d, c_d)
        c = c_d
        deallocate(a_d, b_d, c_d)
    end subroutine suma_vector_gpu

    ! ------------------------------------------------------------
    ! CPU subroutine for vectors (verification)
    ! ------------------------------------------------------------
    subroutine suma_vector_cpu(a, b, c)
        real, intent(in)  :: a(:), b(:)
        real, intent(out) :: c(:)
        integer :: i
        do i = 1, size(a)
            c(i) = a(i) + b(i)
        end do
    end subroutine suma_vector_cpu

    ! ------------------------------------------------------------
    ! Helper subroutine for matrices on GPU
    ! ------------------------------------------------------------
    subroutine suma_matriz_gpu(A, B, C, grid, tBlock)
        real, intent(in)  :: A(:,:), B(:,:)
        real, intent(out) :: C(:,:)
        type(dim3), intent(in) :: grid, tBlock
        real, device, allocatable :: A_d(:,:), B_d(:,:), C_d(:,:)
        integer :: m, n
        m = size(A,1); n = size(A,2)
        allocate(A_d(m,n), B_d(m,n), C_d(m,n))
        A_d = A
        B_d = B
        call suma_matriz_kernel<<<grid, tBlock>>>(A_d, B_d, C_d)
        C = C_d
        deallocate(A_d, B_d, C_d)
    end subroutine suma_matriz_gpu

    ! ------------------------------------------------------------
    ! CPU subroutine for matrices (verification)
    ! ------------------------------------------------------------
    subroutine suma_matriz_cpu(A, B, C)
        real, intent(in)  :: A(:,:), B(:,:)
        real, intent(out) :: C(:,:)
        integer :: i, j
        do i = 1, size(A,1)
            do j = 1, size(A,2)
                C(i,j) = A(i,j) + B(i,j)
            end do
        end do
    end subroutine suma_matriz_cpu

end program first_task

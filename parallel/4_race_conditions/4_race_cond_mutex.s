	.file	"4_race_cond_mutex.c"
	.text
	.globl	pings
	.bss
	.align 4
	.type	pings, @object
	.size	pings, 4
pings:
	.zero	4
	.globl	mutex
	.align 32
	.type	mutex, @object
	.size	mutex, 40
mutex:
	.zero	40
	.text
	.globl	_Z7routinePv
	.type	_Z7routinePv, @function
_Z7routinePv:
.LFB26:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movl	$0, -4(%rbp)
	jmp	.L2
.L3:
	leaq	mutex(%rip), %rax         # load address of mutex to rax
	movq	%rax, %rdi				  # move mutex address to rdi @function call
	call	pthread_mutex_lock@PLT    # call mutex lock function
	movl	pings(%rip), %eax         # load pings value to eax
	addl	$1, %eax                  # increment by 1 
	movl	%eax, pings(%rip)         # move mutex address to rdi
	leaq	mutex(%rip), %rax         # call mutex unlock
	movq	%rax, %rdi                # move mutex address to rdi
	call	pthread_mutex_unlock@PLT  #
	addl	$1, -4(%rbp)
.L2:
	cmpl	$99999, -4(%rbp)
	jle	.L3
	movl	$0, %eax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE26:
	.size	_Z7routinePv, .-_Z7routinePv
	.section	.rodata
.LC0:
	.string	"Mutex init failes"
.LC1:
	.string	"Number of pings : %d\n"
.LC2:
	.string	"Mutex destroy failed"
	.text
	.globl	main
	.type	main, @function
main:
.LFB27:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, %esi
	leaq	mutex(%rip), %rax
	movq	%rax, %rdi
	call	pthread_mutex_init@PLT
	testl	%eax, %eax
	setne	%al
	testb	%al, %al
	je	.L6
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
.L6:
	leaq	-24(%rbp), %rax
	movl	$0, %ecx
	leaq	_Z7routinePv(%rip), %rdx
	movl	$0, %esi
	movq	%rax, %rdi
	call	pthread_create@PLT
	testl	%eax, %eax
	setne	%al
	testb	%al, %al
	je	.L7
	movl	$1, %eax
	jmp	.L13
.L7:
	leaq	-16(%rbp), %rax
	movl	$0, %ecx
	leaq	_Z7routinePv(%rip), %rdx
	movl	$0, %esi
	movq	%rax, %rdi
	call	pthread_create@PLT
	testl	%eax, %eax
	setne	%al
	testb	%al, %al
	je	.L9
	movl	$2, %eax
	jmp	.L13
.L9:
	movq	-24(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	pthread_join@PLT
	testl	%eax, %eax
	setne	%al
	testb	%al, %al
	je	.L10
	movl	$3, %eax
	jmp	.L13
.L10:
	movq	-16(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	pthread_join@PLT
	testl	%eax, %eax
	setne	%al
	testb	%al, %al
	je	.L11
	movl	$4, %eax
	jmp	.L13
.L11:
	movl	pings(%rip), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	mutex(%rip), %rax
	movq	%rax, %rdi
	call	pthread_mutex_destroy@PLT
	testl	%eax, %eax
	setne	%al
	testb	%al, %al
	je	.L12
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
.L12:
	movl	$0, %eax
.L13:
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L14
	call	__stack_chk_fail@PLT
.L14:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE27:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:

#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/IRBuilder.h"


using namespace llvm;

//namespace {

// struct SkeletonPass : public PassInfoMixin<SkeletonPass> {
//     PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {
//         for (auto &F : M) {
//             errs() << "I saw a function called " << F.getName() << "!\n";
//             errs() << "Function body: " << F << "\n";
//             for (auto &B : F){
//                 errs() << "Basic block: " << B << "\n";
//                 for (auto &I : B){
//                     errs() << "Instruction: " << I << "\n";
//                     if (I.getOpcode() == Instruction::Add){
//                         errs() << "I saw an add instruction\n";
//                     }
//                 }
//             }
//         }
//         return PreservedAnalyses::all();
//     };
// };
// }

namespace {
struct SkeletonPass : public PassInfoMixin<SkeletonPass> {
    PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM) {

        for (auto &F : M) {
            LLVMContext& Ctx = F.getContext();
            auto logFunc = F.getParent()->getOrInsertFunction(
                "loggingop", Type::getVoidTy(Ctx), Type::getInt32Ty(Ctx)
            );

            for (auto &B : F){
                for (auto &I : B){
                    if (auto *op = dyn_cast<BinaryOperator>(&I)){
                        IRBuilder<> builder(op);
                        builder.SetInsertPoint(&B, ++builder.GetInsertPoint());
                        Value* args = {op};
                        builder.CreateCall(logFunc, args);
                        // value is base class for operand, basic block is also a value
                        Value *lhs = op->getOperand(0);                        
                        Value *rhs = op->getOperand(1);
                        Value *mul = builder.CreateMul(lhs, rhs);

                        // this loop will replace the first add  operator with the mul operator
                        for (auto& U : op->uses()){
                            User* user = U.getUser();
                            user->setOperand(U.getOperandNo(), mul);
                            errs()<<*op <<"\n"; 
                    }
                    //return PreservedAnalyses::none(); // abort after replacing add
                }
            }
        }
        return PreservedAnalyses::none();
    }
    }
};
}




// this is the entry point for the plugin
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {
        .APIVersion = LLVM_PLUGIN_API_VERSION,
        .PluginName = "Skeleton pass",
        .PluginVersion = "v0.1",
        .RegisterPassBuilderCallbacks = [](PassBuilder &PB) {
            PB.registerPipelineStartEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel Level) {
                    MPM.addPass(SkeletonPass());
                });
        }
    };
}

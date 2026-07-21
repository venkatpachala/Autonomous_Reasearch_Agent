from src.agents.memory_manager import memory_manager
print('Memory Manager initialized successfully!')
print('Chroma stats:', memory_manager.vector.get_collection_stats())
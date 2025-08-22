package com.example.app

import android.annotation.SuppressLint
import android.content.Context
import com.google.ai.edge.localagents.rag.memory.VectorStore
import com.google.ai.edge.localagents.rag.memory.VectorStoreRecord
import com.google.common.collect.ImmutableList
import com.google.common.collect.ImmutableMap
import io.objectbox.Box
import io.objectbox.BoxStore
import io.objectbox.annotation.Entity
import io.objectbox.annotation.HnswIndex
import io.objectbox.annotation.Id
import io.objectbox.annotation.VectorDistanceType
import java.io.File
import kotlin.collections.toFloatArray
import android.util.Log

class ObjectBoxStore(androidContext: Context, directory: File) : VectorStore<String> {
    init {
        Log.i("mam-ai", "LOOK HERE!" + directory)
    }

    val store: BoxStore = MyObjectBox.builder()
        .androidContext(androidContext)
        .directory(directory)
        .build()
    val box: Box<Chunk> = store.boxFor(Chunk::class.java)

    override fun insert(record: VectorStoreRecord<String?>?) {
        TODO("Not yet implemented")
    }

    @SuppressLint("NewApi")
    override fun getNearestRecords(
        queryEmbeddings: List<Float>?,
        topK: Int,
        minSimilarityScore: Float
    ): List<VectorStoreRecord<String?>?>? {
        val arr = (queryEmbeddings!!).toFloatArray()

        return box
            .query(Chunk_.embeddings.nearestNeighbors(arr, topK))
            .build()
            .findWithScores()
            .stream()
            .map { i -> ChunkRecord(i.get()) }
            .toList()
    }
}

@Entity
data class Chunk(
    @Id
    var id: Long = 0,
    var text: String? = null,
    var title: String? = null,
    var page: Int = 0,

    @HnswIndex(dimensions = 768, distanceType = VectorDistanceType.COSINE)
    var embeddings: FloatArray? = null
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as Chunk

        if (id != other.id) return false
        if (page != other.page) return false
        if (text != other.text) return false
        if (title != other.title) return false
        if (!embeddings.contentEquals(other.embeddings)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = id.hashCode()
        result = 31 * result + page
        result = 31 * result + (text?.hashCode() ?: 0)
        result = 31 * result + (title?.hashCode() ?: 0)
        result = 31 * result + (embeddings?.contentHashCode() ?: 0)
        return result
    }
}

class ChunkRecord(val chunk: Chunk): VectorStoreRecord<String?>() {
    override fun getData(): String? = chunk.text

    override fun getEmbeddings(): ImmutableList<Float?>? {
        return if (chunk.embeddings == null) {
            null
        } else {
            ImmutableList.copyOf(chunk.embeddings!!.toTypedArray())
        }
    }

    override fun getMetadata(): ImmutableMap<String?, in Any>? {
        return ImmutableMap.of(
            "title", chunk.title as Any,
            "page", chunk.page as Any
        )
    }

    override fun toBuilder(): Builder<String?>? {
        return ChunkRecordBuilder()
            .setEmbeddings(ImmutableList.copyOf(chunk.embeddings!!.toTypedArray()))!!
            .setData(chunk.text)
    }

    class ChunkRecordBuilder(): Builder<String?>() {
        var text: String? = null
        var embeddings: FloatArray? = null
        var page: Int = 0
        var title: String? = null

        override fun metadataBuilder(): ImmutableMap.Builder<String?, in Any>? {
            return ImmutableMap.builder()
        }

        override fun setData(data: String?): Builder<String?>? {
            text = data
            return this
        }

        override fun setEmbeddings(embeddings: ImmutableList<Float>?): Builder<String?>? {
            this.embeddings = embeddings?.toTypedArray()?.toFloatArray()
            return this
        }

        override fun setMetadata(metadata: ImmutableMap<String?, in Any>?): Builder<String?>? {
            if (metadata?.contains("title") == true) {
                this.title = metadata.get("title") as String
            }

            if (metadata?.contains("page") == true) {
                this.page = metadata.get("page") as Int
            }

            return this
        }

        override fun build(): VectorStoreRecord<String?>? {
            val chunk = Chunk(
                text = text,
                embeddings = embeddings,
                page = page,
                title = title
            )
            return ChunkRecord(chunk) as VectorStoreRecord<String?>?
        }

    }
}
